module systolic_array_4x4 (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        clear_acc,
    input  logic        compute_en,
    input  logic [7:0]  a_data [0:3],
    input  logic [7:0]  b_data [0:3],
    output logic [17:0] c_data [0:15]
);

    logic [7:0] a_bus [0:3][0:4];
    logic [7:0] b_bus [0:4][0:3];

    always_comb begin
        for (int row = 0; row < 4; row++) begin
            a_bus[row][0] = a_data[row];
        end
        for (int col = 0; col < 4; col++) begin
            b_bus[0][col] = b_data[col];
        end
    end

    genvar row, col;
    generate
        for (row = 0; row < 4; row++) begin
            for (col = 0; col < 4; col++) begin : pe_grid
                pe u_pe (
                    .clk        (clk),
                    .rst_n      (rst_n),
                    .clear_acc  (clear_acc),
                    .compute_en (compute_en),
                    .a_in       (a_bus[row][col]),
                    .b_in       (b_bus[row][col]),
                    .a_out      (a_bus[row][col+1]),
                    .b_out      (b_bus[row+1][col]),
                    .acc_out    (c_data[row*4 + col])
                );
            end
        end
    endgenerate

endmodule
