module mma_top_tb;

    logic        clk;
    logic        rst_n;
    logic        start;
    logic [7:0]  data_in;
    logic        data_sel;
    logic        data_valid;
    logic        data_ready;
    logic        out_ready;
    logic [17:0] out_data;
    logic        out_valid;
    logic        out_done;

    logic [7:0] a_mat [0:3][0:3];
    logic [7:0] b_mat [0:3][0:3];
    logic [17:0] expected [0:15];
    logic [17:0] observed [0:15];

    int out_count;
    logic [17:0] stall_data;
    logic        stall_active;
    int          io_count;

    initial begin : fsdb_dump
        $fsdbDumpfile("mma_top_tb.fsdb");
        $fsdbDumpvars(0, mma_top_tb);
        $fsdbDumpMDA();
    end

    mma_top dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (start),
        .data_in   (data_in),
        .data_sel  (data_sel),
        .data_valid(data_valid),
        .data_ready(data_ready),
        .out_ready (out_ready),
        .out_data  (out_data),
        .out_valid (out_valid),
        .out_done  (out_done)
    );

    always #5 clk = ~clk;

    task automatic send_element(
        input logic       select_b,
        input logic [7:0] value
    );
        data_sel = select_b;
        @(negedge clk);
        while (!data_ready) begin
            @(negedge clk);
        end
        data_valid = 1'b1;
        data_in    = value;
        @(negedge clk);
        data_valid = 1'b0;
        data_sel   = 1'b0;
        data_in    = 8'd0;
    endtask

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            io_count <= 0;
        end
        else begin
            io_count <= io_count + 1;
        end
    end

    always_comb begin
        if (!rst_n) begin
            out_ready = 1'b0;
        end
        else begin
            out_ready = (io_count[1:0] != 2'b01);
        end
    end
    
    initial begin
        clk        = 1'b0;
        rst_n      = 1'b0;
        start      = 1'b0;
        data_in    = 8'd0;
        data_sel   = 1'b0;
        data_valid = 1'b0;

        a_mat[0][0] = 8'd1;  a_mat[0][1] = 8'd2;  a_mat[0][2] = 8'd3;  a_mat[0][3] = 8'd4;
        a_mat[1][0] = 8'd5;  a_mat[1][1] = 8'd6;  a_mat[1][2] = 8'd7;  a_mat[1][3] = 8'd8;
        a_mat[2][0] = 8'd9;  a_mat[2][1] = 8'd10; a_mat[2][2] = 8'd11; a_mat[2][3] = 8'd12;
        a_mat[3][0] = 8'd13; a_mat[3][1] = 8'd14; a_mat[3][2] = 8'd15; a_mat[3][3] = 8'd16;

        b_mat[0][0] = 8'd1;  b_mat[0][1] = 8'd0;  b_mat[0][2] = 8'd2;  b_mat[0][3] = 8'd1;
        b_mat[1][0] = 8'd0;  b_mat[1][1] = 8'd1;  b_mat[1][2] = 8'd3;  b_mat[1][3] = 8'd2;
        b_mat[2][0] = 8'd4;  b_mat[2][1] = 8'd1;  b_mat[2][2] = 8'd0;  b_mat[2][3] = 8'd3;
        b_mat[3][0] = 8'd2;  b_mat[3][1] = 8'd5;  b_mat[3][2] = 8'd1;  b_mat[3][3] = 8'd0;

        expected[0]  = 18'd21;  expected[1]  = 18'd25;  expected[2]  = 18'd12;  expected[3]  = 18'd14;
        expected[4]  = 18'd49;  expected[5]  = 18'd53;  expected[6]  = 18'd36;  expected[7]  = 18'd38;
        expected[8]  = 18'd77;  expected[9]  = 18'd81;  expected[10] = 18'd60;  expected[11] = 18'd62;
        expected[12] = 18'd105; expected[13] = 18'd109; expected[14] = 18'd84;  expected[15] = 18'd86;

        repeat (4) @(negedge clk);
        rst_n = 1'b1;

        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        for (int r = 0; r < 4; r++) begin
            for (int c = 0; c < 4; c++) begin
                send_element(1'b0, a_mat[r][c]);
                send_element(1'b1, b_mat[r][c]);
            end
        end

        wait (out_done);
        repeat (2) @(negedge clk);

        if (out_count != 16) begin
            $error("Expected 16 outputs, got %0d", out_count);
        end

        for (int i = 0; i < 16; i++) begin
            if (observed[i] !== expected[i]) begin
                $error("Mismatch at C[%0d]: observed=%0d expected=%0d", i, observed[i], expected[i]);
            end else begin
                $display("C[%0d] match!", i);
            end
        end

        $display("mma_top_tb passed");
        
        $finish;
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            out_count <= 0;
            stall_active <= 1'b0;
            stall_data   <= '0;
            for (int i = 0; i < 16; i++) begin
                observed[i] <= '0;
            end
        end
        else begin
            if (out_valid && !out_ready) begin
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

            if (out_valid && out_ready) begin
                if (out_count < 16) begin
                    observed[out_count] <= out_data;
                    out_count <= out_count + 1;
                end
            end
        end
    end

endmodule
