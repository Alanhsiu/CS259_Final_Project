module mma_top (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  logic [7:0]  data_in,
    input  logic        data_sel,
    input  logic        data_valid,
    output logic        data_ready,
    input  logic        out_ready,
    output logic signed [19:0] out_data,
    output logic        out_valid,
    output logic        out_done
);

    logic       load_en;
    logic [2:0] load_row;
    logic [2:0] load_col;
    logic [3:0] cycle_count;
    logic       feed_en;
    logic       clear_acc;
    logic       compute_en;
    logic       compute_done;
    logic       output_en;

    logic [7:0]  a_data [0:7];
    logic [7:0]  b_data [0:7];
    logic signed [19:0] c_data [0:63];

    controller_fsm u_controller_fsm (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (start),
        .data_sel    (data_sel),
        .data_valid  (data_valid),
        .out_done    (out_done),
        .load_en     (load_en),
        .load_row    (load_row),
        .load_col    (load_col),
        .data_ready  (data_ready),
        .cycle_count (cycle_count),
        .feed_en     (feed_en),
        .clear_acc   (clear_acc),
        .compute_en  (compute_en),
        .compute_done(compute_done),
        .output_en   (output_en)
    );

    buffer_bank u_buffer_bank (
        .clk        (clk),
        .rst_n      (rst_n),
        .load_en    (load_en),
        .load_sel   (data_sel),
        .load_row   (load_row),
        .load_col   (load_col),
        .load_data  (data_in),
        .cycle_count(cycle_count),
        .feed_en    (feed_en),
        .a_data     (a_data),
        .b_data     (b_data)
    );

    systolic_array_8x8 u_systolic_array_8x8 (
        .clk        (clk),
        .rst_n      (rst_n),
        .clear_acc  (clear_acc),
        .compute_en (compute_en),
        .a_data     (a_data),
        .b_data     (b_data),
        .c_data     (c_data)
    );

    buffer_output u_buffer_output (
        .clk        (clk),
        .rst_n      (rst_n),
        .capture_en (output_en),
        .c_data     (c_data),
        .out_ready  (out_ready),
        .out_data   (out_data),
        .out_valid  (out_valid),
        .out_done   (out_done)
    );

endmodule
