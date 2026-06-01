module pe (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        clear_acc,  // Signal to clear the PE at the start of a new computation phase.
    input  logic        compute_en, // Signal to enable computation in the PE.
    input  logic [7:0]  a_in,
    input  logic [7:0]  b_in,
    output logic [7:0]  a_out,
    output logic [7:0]  b_out,
    output logic [17:0] acc_out     // Partial sum.
);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_out <= 18'd0;
            a_out   <= 8'd0;
            b_out   <= 8'd0;
        end
        else begin
            a_out <= a_in;
            b_out <= b_in;

            if (clear_acc) begin
                acc_out <= 18'd0;
            end
            else if (compute_en) begin
                acc_out <= acc_out + (a_in * b_in);
            end
        end
    end

endmodule
