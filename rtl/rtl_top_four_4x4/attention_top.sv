// attention_top.sv  (four 4×4 systolic arrays variant)
// Single-head self-attention: Out = softmax( Q·Kᵀ >>> SCALE_BITS ) · V
//
// The 8×8 output of each matrix multiply is partitioned into four 4×4 quadrants,
// each computed by a dedicated systolic_array_4x4 running in parallel:
//
//   SA00 → C[0:4 , 0:4 ]    SA01 → C[0:4 , 4:8 ]
//   SA10 → C[4:8 , 0:4 ]    SA11 → C[4:8 , 4:8 ]
//
// All four arrays share the same clear/compute controls and cycle counter.
// Each matmul phase completes in COMP_DONE+1 = 14 cycles (vs 22 for 8×8).
//
// Interface is identical to the original attention_top; only internal structure changed.

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

    localparam int NHALF = N / 2;  // 4: each sub-array covers half the rows/cols

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
    logic signed [31:0] accum_rf [0:N-1][0:N-1];  // 32-bit for tiled accumulation
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
    // Four 4×4 systolic arrays (all share clear_acc / compute_en)
    //
    //  a_top  [0:3] = Q rows 0..3  (fed to SA00 and SA01)
    //  a_bot  [0:3] = Q rows 4..7  (fed to SA10 and SA11)
    //  b_left [0:3] = KT/V cols 0..3  (fed to SA00 and SA10)
    //  b_right[0:3] = KT/V cols 4..7  (fed to SA01 and SA11)
    //
    //  Local row/col index r' uses skew delay r' (not r'+NHALF) so wavefronts
    //  inside each sub-array align correctly regardless of global row offset.
    // ---------------------------------------------------------------
    logic [7:0] a_top  [0:NHALF-1];
    logic [7:0] a_bot  [0:NHALF-1];
    logic [7:0] b_left [0:NHALF-1];
    logic [7:0] b_right[0:NHALF-1];

    logic signed [WIN-1:0] c_data_00 [0:NHALF*NHALF-1];
    logic signed [WIN-1:0] c_data_01 [0:NHALF*NHALF-1];
    logic signed [WIN-1:0] c_data_10 [0:NHALF*NHALF-1];
    logic signed [WIN-1:0] c_data_11 [0:NHALF*NHALF-1];

    systolic_array_4x4 u_sa00 (
        .clk(clk), .rst_n(rst_n),
        .clear_acc(clear_acc), .compute_en(compute_en),
        .a_data(a_top),  .b_data(b_left),  .c_data(c_data_00)
    );
    systolic_array_4x4 u_sa01 (
        .clk(clk), .rst_n(rst_n),
        .clear_acc(clear_acc), .compute_en(compute_en),
        .a_data(a_top),  .b_data(b_right), .c_data(c_data_01)
    );
    systolic_array_4x4 u_sa10 (
        .clk(clk), .rst_n(rst_n),
        .clear_acc(clear_acc), .compute_en(compute_en),
        .a_data(a_bot),  .b_data(b_left),  .c_data(c_data_10)
    );
    systolic_array_4x4 u_sa11 (
        .clk(clk), .rst_n(rst_n),
        .clear_acc(clear_acc), .compute_en(compute_en),
        .a_data(a_bot),  .b_data(b_right), .c_data(c_data_11)
    );

    // ---------------------------------------------------------------
    // Input mux (skewed wavefront)
    // For row r' (local 0..NHALF-1):
    //   k = cycle_cnt - r'  must be in 0..N-1
    // a_bot uses global row r'+NHALF but the same local skew delay r'.
    // b_right uses global col c'+NHALF but the same local skew delay c'.
    // ---------------------------------------------------------------
    always_comb begin
        for (int r = 0; r < NHALF; r++) begin
            a_top[r] = 8'd0;
            a_bot[r] = 8'd0;
        end
        for (int c = 0; c < NHALF; c++) begin
            b_left[c]  = 8'd0;
            b_right[c] = 8'd0;
        end

        if (feed_en) begin
            if (mul1_phase) begin
                // Phase 1: Q · KT
                for (int r = 0; r < NHALF; r++) begin
                    if (cycle_cnt >= 5'(r) && (cycle_cnt - 5'(r)) < 5'(N)) begin
                        a_top[r] = Q_buf[r]        [cycle_cnt - r];
                        a_bot[r] = Q_buf[r + NHALF] [cycle_cnt - r];
                    end
                end
                for (int c = 0; c < NHALF; c++) begin
                    if (cycle_cnt >= 5'(c) && (cycle_cnt - 5'(c)) < 5'(N)) begin
                        b_left[c]  = KT_buf[cycle_cnt - c][c];
                        b_right[c] = KT_buf[cycle_cnt - c][c + NHALF];
                    end
                end
            end else begin
                // Phase 3: attn_q07 · V
                for (int r = 0; r < NHALF; r++) begin
                    if (cycle_cnt >= 5'(r) && (cycle_cnt - 5'(r)) < 5'(N)) begin
                        a_top[r] = attn_q07[r]        [cycle_cnt - r];
                        a_bot[r] = attn_q07[r + NHALF] [cycle_cnt - r];
                    end
                end
                for (int c = 0; c < NHALF; c++) begin
                    if (cycle_cnt >= 5'(c) && (cycle_cnt - 5'(c)) < 5'(N)) begin
                        b_left[c]  = V_buf[cycle_cnt - c][c];
                        b_right[c] = V_buf[cycle_cnt - c][c + NHALF];
                    end
                end
            end
        end
    end

    // ---------------------------------------------------------------
    // Tile accumulator: combines all four sub-array outputs into accum_rf
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
            for (int r = 0; r < NHALF; r++)
                for (int c = 0; c < NHALF; c++) begin
                    accum_rf[r]        [c]        <= accum_rf[r]        [c]        + $signed(c_data_00[r*NHALF + c]);
                    accum_rf[r]        [c + NHALF] <= accum_rf[r]        [c + NHALF] + $signed(c_data_01[r*NHALF + c]);
                    accum_rf[r + NHALF][c]        <= accum_rf[r + NHALF][c]        + $signed(c_data_10[r*NHALF + c]);
                    accum_rf[r + NHALF][c + NHALF] <= accum_rf[r + NHALF][c + NHALF] + $signed(c_data_11[r*NHALF + c]);
                end
        end
    end

    // ---------------------------------------------------------------
    // Capture score_rf (accum_rf >>> SCALE_BITS after all tiles)
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
    // Softmax unit (row-wise, N=8, unchanged from original)
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
    // Assemble 64-element c_combined for buffer_output
    // SA00 → [r*N+c],  SA01 → [r*N+(c+NHALF)]
    // SA10 → [(r+NHALF)*N+c],  SA11 → [(r+NHALF)*N+(c+NHALF)]
    // ---------------------------------------------------------------
    logic signed [WIN-1:0] c_combined [0:N*N-1];

    always_comb begin
        for (int r = 0; r < NHALF; r++)
            for (int c = 0; c < NHALF; c++) begin
                c_combined[r*N + c]                    = c_data_00[r*NHALF + c];
                c_combined[r*N + (c + NHALF)]          = c_data_01[r*NHALF + c];
                c_combined[(r + NHALF)*N + c]          = c_data_10[r*NHALF + c];
                c_combined[(r + NHALF)*N + (c + NHALF)] = c_data_11[r*NHALF + c];
            end
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
