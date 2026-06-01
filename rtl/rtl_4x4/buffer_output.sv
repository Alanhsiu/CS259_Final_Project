module buffer_output (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        capture_en,
    input  logic [17:0] c_data [0:15],
    input  logic        out_ready,
    output logic [17:0] out_data,
    output logic        out_valid,
    output logic        out_done
);

    logic [17:0] out_rf [0:15];
    logic [3:0]  rd_idx;

    integer i;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            rd_idx    <= 4'd0;
            out_done  <= 1'b0;
            for (i = 0; i < 16; i++) begin
                out_rf[i] <= 18'd0;
            end
        end
        else begin
            out_done <= 1'b0;

            if (capture_en && !out_valid) begin
                for (i = 0; i < 16; i++) begin
                    out_rf[i] <= c_data[i];
                end
                out_valid <= 1'b1;
                rd_idx    <= 4'd0;
            end
            else if (out_valid && out_ready) begin
                if (rd_idx == 4'd15) begin
                    out_valid <= 1'b0;
                    out_done  <= 1'b1;
                end
                else begin
                    rd_idx <= rd_idx + 4'd1;
                end
            end
        end
    end

    always_comb begin
        if (out_valid) begin
            out_data = out_rf[rd_idx];
        end
        else begin
            out_data = 18'd0;
        end
    end

endmodule
