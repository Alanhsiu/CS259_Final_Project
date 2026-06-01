module controller_fsm (
    input  logic       clk,
    input  logic       rst_n,
    input  logic       start,
    input  logic       data_sel,
    input  logic       data_valid,
    input  logic       out_done,
    output logic        load_en,
    output logic [3:0]  load_row,
    output logic [3:0]  load_col,
    output logic        data_ready,
    output logic [4:0]  cycle_count,
    output logic        feed_en,
    output logic        clear_acc,
    output logic        compute_en,
    output logic        compute_done,
    output logic        output_en
);

    typedef enum logic [2:0] {
        IDLE,
        LOAD,
        CLEAR,
        COMPUTE,
        OUTPUT
    } state_t;

    state_t state, next_state;

    logic [8:0] a_count;       // 0..256
    logic [8:0] b_count;
    logic [5:0] compute_count;  // 0..45
    logic       accept_input;
    logic [7:0] active_count;   // lower 8 bits of a_count/b_count

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
        end
        else begin
            state <= next_state;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_count <= 9'd0;
            b_count <= 9'd0;
        end
        else if (state == IDLE) begin
            a_count <= 9'd0;
            b_count <= 9'd0;
        end
        else if (accept_input) begin
            if (!data_sel) begin
                if (a_count < 9'd256)
                    a_count <= a_count + 9'd1;
            end
            else begin
                if (b_count < 9'd256)
                    b_count <= b_count + 9'd1;
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            compute_count <= 6'd0;
        end
        else if (state == IDLE || state == CLEAR) begin
            compute_count <= 6'd0;
        end
        else if (state == COMPUTE) begin
            if (compute_count < 6'd45)
                compute_count <= compute_count + 6'd1;
        end
    end

    always_comb begin
        active_count = data_sel ? b_count[7:0] : a_count[7:0];
        accept_input = 1'b0;
        load_en      = 1'b0;
        load_row     = active_count[7:4];
        load_col     = active_count[3:0];
        data_ready   = 1'b0;
        cycle_count  = compute_count[4:0];
        feed_en      = 1'b0;
        clear_acc    = 1'b0;
        compute_en   = 1'b0;
        compute_done = 1'b0;
        output_en    = 1'b0;
        next_state   = state;

        case (state)
            IDLE: begin
                if (start)
                    next_state = LOAD;
            end

            LOAD: begin
                data_ready   = data_sel ? (b_count < 9'd256) : (a_count < 9'd256);
                accept_input = data_valid && data_ready;
                load_en      = accept_input;
                if ((a_count == 9'd256) && (b_count == 9'd256))
                    next_state = CLEAR;
            end

            CLEAR: begin
                clear_acc  = 1'b1;
                next_state = COMPUTE;
            end

            COMPUTE: begin
                feed_en    = (compute_count <= 6'd30);
                compute_en = 1'b1;
                if (compute_count == 6'd45) begin
                    compute_done = 1'b1;
                    next_state   = OUTPUT;
                end
            end

            OUTPUT: begin
                output_en = 1'b1;
                if (out_done)
                    next_state = IDLE;
            end

            default: begin
                next_state = IDLE;
            end
        endcase
    end

endmodule
