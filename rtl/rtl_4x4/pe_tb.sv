module pe_tb;

    logic        clk;
    logic        rst_n;
    logic        clear_acc;
    logic        compute_en;
    logic [7:0]  a_in;
    logic [7:0]  b_in;
    logic [7:0]  a_out;
    logic [7:0]  b_out;
    logic [17:0] acc_out;

    initial begin : fsdb_dump
        $fsdbDumpfile("pe_tb.fsdb");
        $fsdbDumpvars(0, pe_tb);
        $fsdbDumpMDA();
    end

    pe dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .clear_acc (clear_acc),
        .compute_en(compute_en),
        .a_in      (a_in),
        .b_in      (b_in),
        .a_out     (a_out),
        .b_out     (b_out),
        .acc_out   (acc_out)
    );

    always #5 clk = ~clk;

    task automatic check_outputs(
        input logic [7:0]  exp_a_out,
        input logic [7:0]  exp_b_out,
        input logic [17:0] exp_acc_out,
        input string       label
    );
        #1;
        if (a_out !== exp_a_out || b_out !== exp_b_out || acc_out !== exp_acc_out) begin
            $error(
                "%s mismatch: a_out=%0d b_out=%0d acc_out=%0d expected a_out=%0d b_out=%0d acc_out=%0d",
                label,
                a_out,
                b_out,
                acc_out,
                exp_a_out,
                exp_b_out,
                exp_acc_out
            );
        end
    endtask

    initial begin
        clk        = 1'b0;
        rst_n      = 1'b0;
        clear_acc  = 1'b0;
        compute_en = 1'b0;
        a_in       = 8'd0;
        b_in       = 8'd0;

        repeat (2) @(posedge clk);
        check_outputs(8'd0, 8'd0, 18'd0, "Reset");
        rst_n = 1'b1;

        @(negedge clk);
        a_in       = 8'd3;
        b_in       = 8'd4;
        compute_en = 1'b1;
        @(posedge clk);
        check_outputs(8'd3, 8'd4, 18'd12, "Cycle 1");

        @(negedge clk);
        a_in = 8'd2;
        b_in = 8'd5;
        @(posedge clk);
        check_outputs(8'd2, 8'd5, 18'd22, "Cycle 2");

        @(negedge clk);
        clear_acc = 1'b1;
        compute_en = 1'b1;
        a_in      = 8'd9;
        b_in      = 8'd9;
        @(posedge clk);
        check_outputs(8'd9, 8'd9, 18'd0, "Clear");

        @(negedge clk);
        clear_acc  = 1'b0;
        compute_en = 1'b0;
        a_in       = 8'd1;
        b_in       = 8'd1;
        @(posedge clk);
        check_outputs(8'd1, 8'd1, 18'd0, "Compute Disabled");

        $display("pe_tb passed");
        $finish;
    end

endmodule
