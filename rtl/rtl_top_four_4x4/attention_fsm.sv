// attention_fsm.sv
// Master sequencer for the tiled two-matmul attention pipeline.
//
// N_ARRAY: systolic array side length (N/2 = 4 for four-4x4 variant).
// COMP_DONE = 2*N_ARRAY + N - 3   (last compute cycle index, was 21, now 13)
// FEED_LAST = N_ARRAY + N - 2     (last feed cycle index,    was 14, now 10)
//
// State sequence (one full computation):
//   IDLE → MUL1_CLR → MUL1_COMP → TILE_CAP →
//     (→ IDLE × NTILES-1)
//     → SCORE_CAP → SOFT_START → SOFT_WAIT (×N rows) →
//   MUL2_CLR → MUL2_COMP → RESULT_CAP → STREAM → DONE_ST → IDLE

module attention_fsm #(
    parameter int N          = 8,
    parameter int NTILES     = 8,
    parameter int N_ARRAY    = 4    // systolic array side; timing derived from this
)(
    input  logic       clk,
    input  logic       rst_n,
    input  logic       start,
    input  logic       sm_done,
    input  logic       out_done,

    output logic       clear_acc,
    output logic       compute_en,
    output logic [4:0] cycle_cnt,
    output logic       feed_en,
    output logic       mul1_phase,

    output logic       accum_en,
    output logic       tile_rdy,

    output logic       capture_score,

    output logic       sm_start,
    output logic [2:0] sm_row,
    output logic       capture_attn,

    output logic       output_en,
    output logic       done
);

    // Last compute cycle: PE[N_ARRAY-1][N_ARRAY-1] accumulates k=N-1 at t = 2*(N_ARRAY-1)+(N-1)
    localparam int COMP_DONE = 2*N_ARRAY + N - 3;  // = 13 for N_ARRAY=4, N=8
    // Last feed cycle: row/col N_ARRAY-1 needs k=N-1 at t = (N_ARRAY-1)+(N-1)
    localparam int FEED_LAST = N_ARRAY + N - 2;    // = 10 for N_ARRAY=4, N=8

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
    logic [3:0]    tile_cnt;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= IDLE;
        else        state <= next_state;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            compute_count <= 5'd0;
        end else if (state == MUL1_CLR || state == MUL2_CLR) begin
            compute_count <= 5'd0;
        end else if ((state == MUL1_COMP || state == MUL2_COMP) && compute_count < 5'(COMP_DONE)) begin
            compute_count <= compute_count + 5'd1;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)                 tile_cnt <= 4'd0;
        else if (state == DONE_ST)  tile_cnt <= 4'd0;
        else if (state == TILE_CAP) tile_cnt <= tile_cnt + 4'd1;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sm_row_r <= 3'd0;
        end else if (state == SCORE_CAP) begin
            sm_row_r <= 3'd0;
        end else if (state == SOFT_WAIT && sm_done) begin
            sm_row_r <= sm_row_r + 3'd1;
        end
    end

    always_comb begin
        next_state = state;
        case (state)
            IDLE:       if (start)                            next_state = MUL1_CLR;
            MUL1_CLR:                                         next_state = MUL1_COMP;
            MUL1_COMP:  if (compute_count == 5'(COMP_DONE))   next_state = TILE_CAP;
            TILE_CAP:   if (tile_cnt == 4'(NTILES - 1))       next_state = SCORE_CAP;
                        else                                  next_state = IDLE;
            SCORE_CAP:                                         next_state = SOFT_START;
            SOFT_START:                                        next_state = SOFT_WAIT;
            SOFT_WAIT:  if (sm_done) begin
                            if (sm_row_r == 3'(N - 1))         next_state = MUL2_CLR;
                            else                               next_state = SOFT_START;
                        end
            MUL2_CLR:                                         next_state = MUL2_COMP;
            MUL2_COMP:  if (compute_count == 5'(COMP_DONE))   next_state = RESULT_CAP;
            RESULT_CAP:                                        next_state = STREAM;
            STREAM:     if (out_done)                          next_state = DONE_ST;
            DONE_ST:                                           next_state = IDLE;
            default:                                           next_state = IDLE;
        endcase
    end

    assign cycle_cnt     = compute_count;

    assign feed_en       = ((state == MUL1_COMP) || (state == MUL2_COMP))
                           && (compute_count <= 5'(FEED_LAST));

    assign compute_en    = (state == MUL1_COMP) || (state == MUL2_COMP);
    assign clear_acc     = (state == MUL1_CLR)  || (state == MUL2_CLR);
    assign mul1_phase    = (state == MUL1_CLR)  || (state == MUL1_COMP);

    assign accum_en      = (state == TILE_CAP);
    assign tile_rdy      = (state == TILE_CAP) && (tile_cnt < 4'(NTILES - 1));

    assign capture_score = (state == SCORE_CAP);

    assign sm_start      = (state == SOFT_START);
    assign sm_row        = sm_row_r;
    assign capture_attn  = (state == SOFT_WAIT) && sm_done;

    assign output_en     = (state == RESULT_CAP);
    assign done          = (state == DONE_ST);

endmodule
