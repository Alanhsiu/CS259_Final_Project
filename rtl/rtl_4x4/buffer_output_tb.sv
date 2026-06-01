module buffer_output_tb;

    logic        clk;
    logic        rst_n;
    logic        capture_en;
    logic [17:0] c_data [0:15];
    logic        out_ready;
    logic [17:0] out_data;
    logic        out_valid;
    logic        out_done;

    logic [17:0] expected [0:15];
    logic [17:0] observed [0:15];
    logic [17:0] stall_data;
    int          accept_idx;
    logic        stall_active;
    int          ready_count;

    initial begin : fsdb_dump
        $fsdbDumpfile("buffer_output_tb.fsdb");
        $fsdbDumpvars(0, buffer_output_tb);
        $fsdbDumpMDA();
    end

    buffer_output dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .capture_en (capture_en),
        .c_data     (c_data),
        .out_ready  (out_ready),
        .out_data   (out_data),
        .out_valid  (out_valid),
        .out_done   (out_done)
    );

    always #5 clk = ~clk;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accept_idx   <= 0;
            stall_active  <= 1'b0;
            stall_data   <= '0;
            ready_count   <= 0;
            for (int i = 0; i < 16; i++) begin
                observed[i] <= '0;
            end
        end
        else begin
            ready_count <= ready_count + 1;

            if (out_valid && out_ready) begin
                if (accept_idx < 16) begin
                    if (out_data !== expected[accept_idx]) begin
                        $error(
                            "Accepted data mismatch at index %0d: got=%0d expected=%0d",
                            accept_idx,
                            out_data,
                            expected[accept_idx]
                        );
                    end
                    observed[accept_idx] <= out_data;
                    accept_idx <= accept_idx + 1;
                end
            end
            else if (out_valid && !out_ready) begin
                if (!stall_active) begin
                    stall_active <= 1'b1;
                    stall_data   <= out_data;
                end
                else if (out_data !== stall_data) begin
                    $error("out_data changed while out_ready was low");
                end
            end
            else begin
                stall_active <= 1'b0;
            end
        end
    end

    initial begin
        clk        = 1'b0;
        rst_n      = 1'b0;
        capture_en = 1'b0;

        for (int i = 0; i < 16; i++) begin
            c_data[i]   = 18'd0;
            expected[i] = 18'd0;
        end

        for (int i = 0; i < 16; i++) begin
            c_data[i]   = 18'd200 + i;
            expected[i] = 18'd200 + i;
        end

        repeat (2) @(posedge clk);
        rst_n = 1'b1;

        @(negedge clk);
        capture_en = 1'b1;
        @(negedge clk);
        capture_en = 1'b0;

        wait (out_done);
        if (out_done !== 1'b1) begin
            $error("out_done should be high when the final element is accepted");
        end

        repeat (2) @(posedge clk);

        @(posedge clk);
        #1;
        if (out_valid !== 1'b0 || out_done !== 1'b0) begin
            $error("output_buffer should return idle after the transfer");
        end

        if (accept_idx !== 16) begin
            $error("Expected 16 accepted outputs, got %0d", accept_idx);
        end

        for (int i = 0; i < 16; i++) begin
            if (observed[i] !== expected[i]) begin
                $error("Mismatch at index %0d: observed=%0d expected=%0d", i, observed[i], expected[i]);
            end
        end

        $display("output_buffer_tb passed");
        $finish;
    end

    always_comb begin
        if (!rst_n) begin
            out_ready = 1'b0;
        end
        else begin
            out_ready = (ready_count[1:0] != 2'b01);
        end
    end

endmodule
