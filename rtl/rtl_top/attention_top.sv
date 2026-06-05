// attention_top.sv
// Single-head self-attention: Out = softmax( Q·Kᵀ >>> SCALE_BITS ) · V
//
// Supports tiled Q·Kᵀ: the host loads one 8-column KT slice per start pulse.
// After NTILES passes the hardware applies >>>SCALE_BITS (= /sqrt(d_k)) and
// proceeds to softmax then attn·V automatically.
//
// Parameters
//   N          = tile size (seq_len = systolic array side = 8)
//   WIN        = accumulator width (20 bits, signed)
//   WOUT       = softmax output width (Q1.15, 16 bits)
//   NTILES     = number of KT tiles; full d_k = N * NTILES  (default 8 → d_k=64)
//   SCALE_BITS = arithmetic right-shift applied to accumulated Q·Kᵀ score
//                (default 3 → divide by 8 = sqrt(64), exact for d_k=64)
//
// Interface
//   load_sel : 2'b00=Q, 2'b01=Kᵀ (current tile), 2'b10=V
//   Load Q and V once before start.  Reload KT for every tile.
//   Assert start after each KT load.
//   tile_rdy pulses after tiles 0..NTILES-2 (host reloads KT and re-asserts start).
//   done pulses after the final stream (all 64 output words sent).
//
// Output
//   out_data: 20-bit signed; represents (attn · V) element.
//   To recover INT8 range: out_data >>> 7  (Q0.6 quantisation of attn weights).

module attention_top #(
    parameter int N          = 8,
    parameter int WIN        = 20,
    parameter int WOUT       = 16,
    parameter int NTILES     = 8,   // K tiles; full d_k = N * NTILES
    parameter int SCALE_BITS = 3    // >>>3 = /8 = /sqrt(64) for d_k=64
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
    output logic        tile_rdy,   // pulse: tile done, host reloads KT
    output logic        done
);

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
    // accum_rf: 32-bit signed accumulator for tiled Q·Kᵀ (cleared after done)
    logic signed [31:0] accum_rf [0:N-1][0:N-1];

    // score_rf: scaled Q·Kᵀ result (accum_rf >>> SCALE_BITS), captured once after all tiles
    logic signed [WIN-1:0] score_rf [0:N-1][0:N-1];

    // attn_rf: softmax Q1.15 output
    logic [WOUT-1:0] attn_rf [0:N-1][0:N-1];

    // attn_q07: quantised attention weights for PE input (Q0.6, bit[7]=0)
    logic [7:0] attn_q07 [0:N-1][0:N-1];
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

    attention_fsm #(.N(N), .NTILES(NTILES)) u_fsm (
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
    // Systolic array
    // ---------------------------------------------------------------
    logic [7:0]            a_data [0:N-1];
    logic [7:0]            b_data [0:N-1];
    logic signed [WIN-1:0] c_data [0:N*N-1];

    systolic_array_8x8 u_sa (
        .clk       (clk),
        .rst_n     (rst_n),
        .clear_acc (clear_acc),
        .compute_en(compute_en),
        .a_data    (a_data),
        .b_data    (b_data),
        .c_data    (c_data)
    );

    // ---------------------------------------------------------------
    // Input mux (skewed wavefront)
    // ---------------------------------------------------------------
    always_comb begin
        for (int r = 0; r < N; r++) a_data[r] = 8'd0;
        for (int c = 0; c < N; c++) b_data[c] = 8'd0;

        if (feed_en) begin
            if (mul1_phase) begin
                for (int r = 0; r < N; r++) begin
                    if (cycle_cnt >= 5'(r) && (cycle_cnt - 5'(r)) < 5'(N))
                        a_data[r] = Q_buf[r][cycle_cnt - r];
                end
                for (int c = 0; c < N; c++) begin
                    if (cycle_cnt >= 5'(c) && (cycle_cnt - 5'(c)) < 5'(N))
                        b_data[c] = KT_buf[cycle_cnt - c][c];
                end
            end else begin
                for (int r = 0; r < N; r++) begin
                    if (cycle_cnt >= 5'(r) && (cycle_cnt - 5'(r)) < 5'(N))
                        a_data[r] = attn_q07[r][cycle_cnt - r];
                end
                for (int c = 0; c < N; c++) begin
                    if (cycle_cnt >= 5'(c) && (cycle_cnt - 5'(c)) < 5'(N))
                        b_data[c] = V_buf[cycle_cnt - c][c];
                end
            end
        end
    end

    // ---------------------------------------------------------------
    // Tile accumulator
    //   accum_rf[r][c] += c_data[r*N+c]  on each accum_en pulse
    //   Cleared after done (ready for next full computation)
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
            for (int r = 0; r < N; r++)
                for (int c = 0; c < N; c++)
                    accum_rf[r][c] <= accum_rf[r][c] + $signed(c_data[r*N+c]);
        end
    end

    // ---------------------------------------------------------------
    // Capture score_rf (accum_rf >>> SCALE_BITS after all tiles done)
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
    // Softmax unit (row-wise, N=8)
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
    // Capture attn_rf from softmax output
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
    // Output: buffer_output streams attn·V results
    // ---------------------------------------------------------------
    buffer_output u_bout (
        .clk       (clk),
        .rst_n     (rst_n),
        .capture_en(output_en),
        .c_data    (c_data),
        .out_ready (out_ready),
        .out_data  (out_data),
        .out_valid (out_valid),
        .out_done  (out_done_w)
    );

endmodule
