// attention_top.sv  (sixteen 2×2 systolic arrays variant)
// Single-head self-attention: Out = softmax( Q·Kᵀ >>> SCALE_BITS ) · V
//
// The 8×8 output of each matrix multiply is partitioned into sixteen 2×2 blocks.
// Arrays are arranged in a NBLK×NBLK (4×4) grid, each block covering a 2×2 output:
//
//   SA[i][j] → output rows i*2..i*2+1, cols j*2..j*2+1   (i,j ∈ 0..3)
//
// Arrays in the same block-row i share identical a_data (from Q rows i*2..i*2+1).
// Arrays in the same block-col j share identical b_data (from KT/V cols j*2..j*2+1).
// All 16 arrays run in parallel with the same clear/compute controls.
//
// Matmul phase completes in COMP_DONE+1 = 10 cycles (vs 22 / 14 for 8×8 / 4×4).
//
// Interface is identical to the original attention_top.

module attention_top #(
    parameter int N          = 8,
    parameter int WIN        = 20,
    parameter int WOUT       = 16,
    parameter int NTILES     = 8,
    parameter int SCALE_BITS = 3
)(
    input  logic        clk,
    input  logic        rst_n,

    input  logic [1:0]  load_sel,
    input  logic        load_en,
    input  logic [2:0]  load_row,
    input  logic [2:0]  load_col,
    input  logic [7:0]  load_data,

    input  logic        start,

    input  logic        out_ready,
    output logic signed [WIN-1:0] out_data,
    output logic        out_valid,
    output logic        tile_rdy,
    output logic        done
);

    localparam int NHALF = 2;        // each sub-array is NHALF×NHALF = 2×2
    localparam int NBLK  = N / NHALF; // 4 blocks per dimension → 16 arrays total

    // ---------------------------------------------------------------
    // Input matrices
    // ---------------------------------------------------------------
    logic signed [7:0] Q_buf  [0:N-1][0:N-1];
    logic signed [7:0] KT_buf [0:N-1][0:N-1];
    logic signed [7:0] V_buf  [0:N-1][0:N-1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int r = 0; r < N; r++)
                for (int c = 0; c < N; c++) begin
                    Q_buf [r][c] <= 8'sd0;
                    KT_buf[r][c] <= 8'sd0;
                    V_buf [r][c] <= 8'sd0;
                end
        end else if (load_en) begin
            case (load_sel)
                2'b00: Q_buf [load_row][load_col] <= load_data;
                2'b01: KT_buf[load_row][load_col] <= load_data;
                2'b10: V_buf [load_row][load_col] <= load_data;
                default: ;
            endcase
        end
    end

    // ---------------------------------------------------------------
    // Intermediate register files
    // ---------------------------------------------------------------
    logic signed [31:0] accum_rf [0:N-1][0:N-1];
    logic signed [WIN-1:0] score_rf [0:N-1][0:N-1];
    logic [WOUT-1:0] attn_rf  [0:N-1][0:N-1];
    logic [7:0]      attn_q07 [0:N-1][0:N-1];

    generate
        genvar gi, gj;
        for (gi = 0; gi < N; gi++)
            for (gj = 0; gj < N; gj++)
                assign attn_q07[gi][gj] = attn_rf[gi][gj][WOUT-1]
                                          ? 8'h7F
                                          : {1'b0, attn_rf[gi][gj][WOUT-2 : WOUT-8]};
    endgenerate

    // ---------------------------------------------------------------
    // FSM
    // ---------------------------------------------------------------
    logic       clear_acc, compute_en, feed_en, mul1_phase;
    logic [4:0] cycle_cnt;
    logic       accum_en_w;
    logic       capture_score;
    logic       sm_start, capture_attn;
    logic [2:0] sm_row;
    logic       output_en, out_done_w, done_fsm, sm_done_w;

    attention_fsm #(.N(N), .NTILES(NTILES), .N_ARRAY(NHALF)) u_fsm (
        .clk          (clk),
        .rst_n        (rst_n),
        .start        (start),
        .sm_done      (sm_done_w),
        .out_done     (out_done_w),
        .clear_acc    (clear_acc),
        .compute_en   (compute_en),
        .cycle_cnt    (cycle_cnt),
        .feed_en      (feed_en),
        .mul1_phase   (mul1_phase),
        .accum_en     (accum_en_w),
        .tile_rdy     (tile_rdy),
        .capture_score(capture_score),
        .sm_start     (sm_start),
        .sm_row       (sm_row),
        .capture_attn (capture_attn),
        .output_en    (output_en),
        .done         (done_fsm)
    );

    assign done = done_fsm;

    // ---------------------------------------------------------------
    // Sixteen 2×2 systolic arrays, arranged as a NBLK×NBLK (4×4) grid.
    //
    //  a_row[i][r'] = Q / attn_q07 rows i*NHALF .. i*NHALF+NHALF-1
    //                 shared by all NBLK arrays in block-row i
    //  b_col[j][c'] = KT / V cols j*NHALF .. j*NHALF+NHALF-1
    //                 shared by all NBLK arrays in block-col j
    //
    //  Skew delay is based on LOCAL index r' / c' (not global row/col),
    //  matching the wavefront alignment inside each sub-array.
    // ---------------------------------------------------------------
    logic [7:0] a_row [0:NBLK-1][0:NHALF-1];
    logic [7:0] b_col [0:NBLK-1][0:NHALF-1];

    logic signed [WIN-1:0] c_data_arr [0:NBLK-1][0:NBLK-1][0:NHALF*NHALF-1];

    genvar brow, bcol;
    generate
        for (brow = 0; brow < NBLK; brow++) begin : row_blk
            for (bcol = 0; bcol < NBLK; bcol++) begin : col_blk
                systolic_array_2x2 u_sa (
                    .clk        (clk),
                    .rst_n      (rst_n),
                    .clear_acc  (clear_acc),
                    .compute_en (compute_en),
                    .a_data     (a_row[brow]),
                    .b_data     (b_col[bcol]),
                    .c_data     (c_data_arr[brow][bcol])
                );
            end
        end
    endgenerate

    // ---------------------------------------------------------------
    // Input mux (skewed wavefront)
    // For local index r' in 0..NHALF-1:
    //   k = cycle_cnt - r'  must be in 0..N-1
    // Global row = i*NHALF + r', but skew delay is r' (local position).
    // ---------------------------------------------------------------
    always_comb begin
        for (int i = 0; i < NBLK; i++)
            for (int r = 0; r < NHALF; r++)
                a_row[i][r] = 8'd0;
        for (int j = 0; j < NBLK; j++)
            for (int c = 0; c < NHALF; c++)
                b_col[j][c] = 8'd0;

        if (feed_en) begin
            if (mul1_phase) begin
                // Phase 1: Q · KT
                for (int i = 0; i < NBLK; i++)
                    for (int r = 0; r < NHALF; r++)
                        if (cycle_cnt >= 5'(r) && (cycle_cnt - 5'(r)) < 5'(N))
                            a_row[i][r] = Q_buf[i*NHALF + r][cycle_cnt - r];
                for (int j = 0; j < NBLK; j++)
                    for (int c = 0; c < NHALF; c++)
                        if (cycle_cnt >= 5'(c) && (cycle_cnt - 5'(c)) < 5'(N))
                            b_col[j][c] = KT_buf[cycle_cnt - c][j*NHALF + c];
            end else begin
                // Phase 3: attn_q07 · V
                for (int i = 0; i < NBLK; i++)
                    for (int r = 0; r < NHALF; r++)
                        if (cycle_cnt >= 5'(r) && (cycle_cnt - 5'(r)) < 5'(N))
                            a_row[i][r] = attn_q07[i*NHALF + r][cycle_cnt - r];
                for (int j = 0; j < NBLK; j++)
                    for (int c = 0; c < NHALF; c++)
                        if (cycle_cnt >= 5'(c) && (cycle_cnt - 5'(c)) < 5'(N))
                            b_col[j][c] = V_buf[cycle_cnt - c][j*NHALF + c];
            end
        end
    end

    // ---------------------------------------------------------------
    // Tile accumulator: maps all 16 sub-array outputs into accum_rf
    // SA[i][j].c_data[r'*NHALF+c'] → accum_rf[i*NHALF+r'][j*NHALF+c']
    // ---------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int r = 0; r < N; r++)
                for (int c = 0; c < N; c++)
                    accum_rf[r][c] <= 32'sd0;
        end else if (done_fsm) begin
            for (int r = 0; r < N; r++)
                for (int c = 0; c < N; c++)
                    accum_rf[r][c] <= 32'sd0;
        end else if (accum_en_w) begin
            for (int i = 0; i < NBLK; i++)
                for (int j = 0; j < NBLK; j++)
                    for (int r = 0; r < NHALF; r++)
                        for (int c = 0; c < NHALF; c++)
                            accum_rf[i*NHALF+r][j*NHALF+c] <=
                                accum_rf[i*NHALF+r][j*NHALF+c] +
                                $signed(c_data_arr[i][j][r*NHALF+c]);
        end
    end

    // ---------------------------------------------------------------
    // Capture score_rf
    // ---------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int r = 0; r < N; r++)
                for (int c = 0; c < N; c++)
                    score_rf[r][c] <= '0;
        end else if (capture_score) begin
            for (int r = 0; r < N; r++)
                for (int c = 0; c < N; c++)
                    score_rf[r][c] <= WIN'(accum_rf[r][c] >>> SCALE_BITS);
        end
    end

    // ---------------------------------------------------------------
    // Softmax unit (row-wise, N=8, unchanged)
    // ---------------------------------------------------------------
    logic [WIN-1:0]  sm_data_in  [N];
    logic [WOUT-1:0] sm_data_out [N];

    always_comb begin
        for (int c = 0; c < N; c++)
            sm_data_in[c] = WIN'(score_rf[sm_row][c]);
    end

    softmax_unit #(.N(N), .WIN(WIN), .WOUT(WOUT)) u_sm (
        .clk     (clk),
        .rst_n   (rst_n),
        .start   (sm_start),
        .data_in (sm_data_in),
        .done    (sm_done_w),
        .data_out(sm_data_out)
    );

    // ---------------------------------------------------------------
    // Capture attn_rf
    // ---------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int r = 0; r < N; r++)
                for (int c = 0; c < N; c++)
                    attn_rf[r][c] <= '0;
        end else if (capture_attn) begin
            for (int c = 0; c < N; c++)
                attn_rf[sm_row][c] <= sm_data_out[c];
        end
    end

    // ---------------------------------------------------------------
    // Assemble c_combined for buffer_output
    // c_combined[(i*NHALF+r)*N + (j*NHALF+c)] = SA[i][j].c_data[r*NHALF+c]
    // ---------------------------------------------------------------
    logic signed [WIN-1:0] c_combined [0:N*N-1];

    always_comb begin
        for (int i = 0; i < NBLK; i++)
            for (int j = 0; j < NBLK; j++)
                for (int r = 0; r < NHALF; r++)
                    for (int c = 0; c < NHALF; c++)
                        c_combined[(i*NHALF+r)*N + (j*NHALF+c)] =
                            c_data_arr[i][j][r*NHALF+c];
    end

    // ---------------------------------------------------------------
    // Output: buffer_output streams attn·V results
    // ---------------------------------------------------------------
    buffer_output u_bout (
        .clk       (clk),
        .rst_n     (rst_n),
        .capture_en(output_en),
        .c_data    (c_combined),
        .out_ready (out_ready),
        .out_data  (out_data),
        .out_valid (out_valid),
        .out_done  (out_done_w)
    );

endmodule
