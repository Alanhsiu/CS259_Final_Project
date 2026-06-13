module systolic_array_2x2 (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        clear_acc,
    input  logic        compute_en,
    input  logic [7:0]  a_data [0:1],
    input  logic [7:0]  b_data [0:1],
    output logic signed [19:0] c_data [0:3]
);

    logic [7:0] a_bus [0:1][0:2];
    logic [7:0] b_bus [0:2][0:1];

    always_comb begin
        for (int row = 0; row < 2; row++)
            a_bus[row][0] = a_data[row];
        for (int col = 0; col < 2; col++)
            b_bus[0][col] = b_data[col];
    end

    genvar row, col;
    generate
        for (row = 0; row < 2; row++) begin
            for (col = 0; col < 2; col++) begin : pe_grid
                pe u_pe (
                    .clk        (clk),
                    .rst_n      (rst_n),
                    .clear_acc  (clear_acc),
                    .compute_en (compute_en),
                    .a_in       (a_bus[row][col]),
                    .b_in       (b_bus[row][col]),
                    .a_out      (a_bus[row][col+1]),
                    .b_out      (b_bus[row+1][col]),
                    .acc_out    (c_data[row*2 + col])
                );
            end
        end
    endgenerate

endmodule
