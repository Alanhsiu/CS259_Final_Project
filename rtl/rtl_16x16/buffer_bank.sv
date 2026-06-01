module buffer_bank (
    input  logic       clk,
    input  logic       rst_n,

    input  logic        load_en,
    input  logic        load_sel,
    input  logic [3:0]  load_row,
    input  logic [3:0]  load_col,
    input  logic [7:0]  load_data,

    input  logic [4:0]  cycle_count,
    input  logic        feed_en,

    output logic [7:0]  a_data [0:15],
    output logic [7:0]  b_data [0:15]
);

    buffer_input u_buffer_input (
        .clk        (clk),
        .rst_n      (rst_n),
        .load_en    (load_en),
        .load_sel   (load_sel),
        .load_row   (load_row),
        .load_col   (load_col),
        .load_data  (load_data),
        .cycle_count(cycle_count),
        .feed_en    (feed_en),
        .a_data     (a_data),
        .b_data     (b_data)
    );

endmodule
