module pe (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        clear_acc,
    input  logic        compute_en,
    input  logic [7:0]  a_in,
    input  logic [7:0]  b_in,
    output logic        [7:0]  a_out,
    output logic        [7:0]  b_out,
    output logic signed [19:0] acc_out
);

    // Sign-extend inputs to 16 bits so the product captures the full range
    // of signed 8-bit × signed 8-bit (-128*-128 = 16384, fits in 16-bit signed).
    logic signed [15:0] a_ext, b_ext, product;

    always_comb begin
        a_ext   = {{8{a_in[7]}}, a_in};
        b_ext   = {{8{b_in[7]}}, b_in};
        product = a_ext * b_ext;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_out <= 20'sd0;
            a_out   <= 8'd0;
            b_out   <= 8'd0;
        end
        else begin
            a_out <= a_in;
            b_out <= b_in;

            if (clear_acc) begin
                acc_out <= 20'sd0;
            end
            else if (compute_en) begin
                // Sign-extend 16-bit product to 20 bits before accumulating.
                acc_out <= $signed(acc_out) + {{4{product[15]}}, product};
            end
        end
    end

endmodule
