// attention_fsm.sv
// Master sequencer for the tiled two-matmul attention pipeline.
//
// Tiling: Phase 1 runs NTILES times (one 8-column K tile per pass).
//   Each pass: MUL1_CLR → MUL1_COMP → TILE_CAP (accum c_data, increment tile_cnt)
//   If more tiles remain → IDLE (host loads next KT slice, re-asserts start)
//   After last tile    → SCORE_CAP (accum_rf >>> SCALE_BITS → score_rf)
//
// Host protocol:
//   1. Load Q, KT_tile[0], V.  Assert start.  Wait for tile_rdy.
//   2. Repeat for tiles 1..NTILES-2: reload KT, assert start, wait tile_rdy.
//   3. Reload KT_tile[NTILES-1], assert start.  Wait for done.
//
// State sequence (one full computation):
//   IDLE → MUL1_CLR → MUL1_COMP → TILE_CAP →
//     (→ IDLE × NTILES-1)
//     → SCORE_CAP → SOFT_START → SOFT_WAIT (×N rows) →
//   MUL2_CLR → MUL2_COMP → RESULT_CAP → STREAM → DONE_ST → IDLE

module attention_fsm #(
    parameter int N          = 8,   // seq_len = tile d_k
    parameter int NTILES     = 8    // K tiles; full d_k = N * NTILES
)(
    input  logic       clk,
    input  logic       rst_n,
    input  logic       start,
    input  logic       sm_done,
    input  logic       out_done,

    // Systolic array controls
    output logic       clear_acc,
    output logic       compute_en,
    output logic [4:0] cycle_cnt,
    output logic       feed_en,
    output logic       mul1_phase,

    // Tiling controls
    output logic       accum_en,    // pulse: add c_data to accum_rf
    output logic       tile_rdy,    // pulse: tile done, more tiles remain; host reloads KT

    // Score capture
    output logic       capture_score,

    // Softmax controls
    output logic       sm_start,
    output logic [2:0] sm_row,
    output logic       capture_attn,

    // Output controls
    output logic       output_en,
    output logic       done
);

    // ---------------------------------------------------------------
    // State encoding
    // ---------------------------------------------------------------
    typedef enum logic [3:0] {
        IDLE,
        MUL1_CLR,
        MUL1_COMP,
        TILE_CAP,
        SCORE_CAP,
        SOFT_START,
        SOFT_WAIT,
        MUL2_CLR,
        MUL2_COMP,
        RESULT_CAP,
        STREAM,
        DONE_ST
    } state_t;

    state_t        state, next_state;
    logic [4:0]    compute_count;
    logic [2:0]    sm_row_r;
    logic [3:0]    tile_cnt;   // 0..NTILES-1

    // ---------------------------------------------------------------
    // State register
    // ---------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= IDLE;
        else        state <= next_state;
    end

    // ---------------------------------------------------------------
    // Compute-cycle counter (shared between MUL1 and MUL2)
    // ---------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            compute_count <= 5'd0;
        end else if (state == MUL1_CLR || state == MUL2_CLR) begin
            compute_count <= 5'd0;
        end else if ((state == MUL1_COMP || state == MUL2_COMP) && compute_count < 5'd21) begin
            compute_count <= compute_count + 5'd1;
        end
    end

    // ---------------------------------------------------------------
    // Tile counter (0..NTILES-1; resets after DONE_ST)
    // ---------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)                 tile_cnt <= 4'd0;
        else if (state == DONE_ST)  tile_cnt <= 4'd0;
        else if (state == TILE_CAP) tile_cnt <= tile_cnt + 4'd1;
    end

    // ---------------------------------------------------------------
    // Softmax row counter
    // ---------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sm_row_r <= 3'd0;
        end else if (state == SCORE_CAP) begin
            sm_row_r <= 3'd0;
        end else if (state == SOFT_WAIT && sm_done) begin
            sm_row_r <= sm_row_r + 3'd1;
        end
    end

    // ---------------------------------------------------------------
    // Next-state logic
    // ---------------------------------------------------------------
    always_comb begin
        next_state = state;
        case (state)
            IDLE:       if (start)                        next_state = MUL1_CLR;

            MUL1_CLR:                                     next_state = MUL1_COMP;

            MUL1_COMP:  if (compute_count == 5'd21)       next_state = TILE_CAP;

            TILE_CAP:   if (tile_cnt == 4'(NTILES - 1))  next_state = SCORE_CAP;
                        else                              next_state = IDLE;

            SCORE_CAP:                                    next_state = SOFT_START;

            SOFT_START:                                   next_state = SOFT_WAIT;

            SOFT_WAIT:  if (sm_done) begin
                            if (sm_row_r == 3'(N - 1))   next_state = MUL2_CLR;
                            else                         next_state = SOFT_START;
                        end

            MUL2_CLR:                                     next_state = MUL2_COMP;

            MUL2_COMP:  if (compute_count == 5'd21)       next_state = RESULT_CAP;

            RESULT_CAP:                                   next_state = STREAM;

            STREAM:     if (out_done)                     next_state = DONE_ST;

            DONE_ST:                                      next_state = IDLE;

            default:                                      next_state = IDLE;
        endcase
    end

    // ---------------------------------------------------------------
    // Output assignments
    // ---------------------------------------------------------------
    assign cycle_cnt     = compute_count;

    assign feed_en       = ((state == MUL1_COMP) || (state == MUL2_COMP))
                           && (compute_count <= 5'(N + N - 2));

    assign compute_en    = (state == MUL1_COMP) || (state == MUL2_COMP);
    assign clear_acc     = (state == MUL1_CLR)  || (state == MUL2_CLR);

    assign mul1_phase    = (state == MUL1_CLR)  || (state == MUL1_COMP);

    // accum_en fires once per tile (in TILE_CAP) to add c_data into accum_rf
    assign accum_en      = (state == TILE_CAP);

    // tile_rdy: more tiles remain; fires in TILE_CAP for tiles 0..NTILES-2
    assign tile_rdy      = (state == TILE_CAP) && (tile_cnt < 4'(NTILES - 1));

    assign capture_score = (state == SCORE_CAP);

    assign sm_start      = (state == SOFT_START);
    assign sm_row        = sm_row_r;
    assign capture_attn  = (state == SOFT_WAIT) && sm_done;

    assign output_en     = (state == RESULT_CAP);
    assign done          = (state == DONE_ST);

endmodule
