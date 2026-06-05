module systolic_array_8x8 (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        clear_acc,
    input  logic        compute_en,
    input  logic [7:0]  a_data [0:7],
    input  logic [7:0]  b_data [0:7],
    output logic signed [19:0] c_data [0:63]
);

    logic [7:0] a_bus [0:7][0:8];
    logic [7:0] b_bus [0:8][0:7];

    always_comb begin
        for (int row = 0; row < 8; row++) begin
            a_bus[row][0] = a_data[row];
        end
        for (int col = 0; col < 8; col++) begin
            b_bus[0][col] = b_data[col];
        end
    end

    genvar row, col;
    generate
        for (row = 0; row < 8; row++) begin
            for (col = 0; col < 8; col++) begin : pe_grid
                pe u_pe (
                    .clk        (clk),
                    .rst_n      (rst_n),
                    .clear_acc  (clear_acc),
                    .compute_en (compute_en),
                    .a_in       (a_bus[row][col]),
                    .b_in       (b_bus[row][col]),
                    .a_out      (a_bus[row][col+1]),
                    .b_out      (b_bus[row+1][col]),
                    .acc_out    (c_data[row*8 + col])
                );
            end
        end
    endgenerate

endmodule
