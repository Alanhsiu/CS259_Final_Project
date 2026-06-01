module buffer_input_tb;

    logic       clk;
    logic       rst_n;
    logic       load_en;
    logic       load_sel;
    logic [1:0] load_row;
    logic [1:0] load_col;
    logic [7:0] load_data;
    logic [2:0] cycle_count;
    logic       feed_en;
    logic [7:0] a_data [0:3];
    logic [7:0] b_data [0:3];

    logic [7:0] a_mat [0:3][0:3];
    logic [7:0] b_mat [0:3][0:3];
    logic [7:0] exp_a [0:3];
    logic [7:0] exp_b [0:3];
    logic [31:0] a_vec;
    logic [31:0] b_vec;
    logic [31:0] exp_a_vec;
    logic [31:0] exp_b_vec;

    initial begin : fsdb_dump
        $fsdbDumpfile("buffer_input_tb.fsdb");
        $fsdbDumpvars(0, buffer_input_tb);
        $fsdbDumpMDA();
    end

    buffer_input dut (
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

    always #5 clk = ~clk;

    always_comb begin
        for (int r = 0; r < 4; r++) begin
            exp_a[r] = 8'd0;
        end
        for (int c = 0; c < 4; c++) begin
            exp_b[c] = 8'd0;
        end

        if (feed_en) begin
            for (int r = 0; r < 4; r++) begin
                if ((cycle_count >= r) && ((cycle_count - r) < 4)) begin
                    exp_a[r] = a_mat[r][cycle_count - r];
                end
            end
            for (int c = 0; c < 4; c++) begin
                if ((cycle_count >= c) && ((cycle_count - c) < 4)) begin
                    exp_b[c] = b_mat[cycle_count - c][c];
                end
            end
        end

        a_vec     = {a_data[0], a_data[1], a_data[2], a_data[3]};
        b_vec     = {b_data[0], b_data[1], b_data[2], b_data[3]};
        exp_a_vec = {exp_a[0], exp_a[1], exp_a[2], exp_a[3]};
        exp_b_vec = {exp_b[0], exp_b[1], exp_b[2], exp_b[3]};
    end

    task automatic write_element(
        input logic       select_b,
        input logic [1:0] row,
        input logic [1:0] col,
        input logic [7:0] value
    );
        @(negedge clk);
        load_en   = 1'b1;
        load_sel  = select_b;
        load_row  = row;
        load_col  = col;
        load_data = value;
        @(negedge clk);
        load_en   = 1'b0;
        load_sel  = 1'b0;
        load_row  = 2'd0;
        load_col  = 2'd0;
        load_data = 8'd0;
    endtask

    initial begin
        clk        = 1'b0;
        rst_n      = 1'b0;
        load_en    = 1'b0;
        load_sel   = 1'b0;
        load_row   = 2'd0;
        load_col   = 2'd0;
        load_data  = 8'd0;
        cycle_count = 3'd0;
        feed_en    = 1'b0;

        for (int r = 0; r < 4; r++) begin
            a_mat[r][0] = 8'd0;
            a_mat[r][1] = 8'd0;
            a_mat[r][2] = 8'd0;
            a_mat[r][3] = 8'd0;
            b_mat[r][0] = 8'd0;
            b_mat[r][1] = 8'd0;
            b_mat[r][2] = 8'd0;
            b_mat[r][3] = 8'd0;
        end

        for (int r = 0; r < 4; r++) begin
            for (int c = 0; c < 4; c++) begin
                a_mat[r][c] = (r * 4 + c + 1);
                b_mat[r][c] = (8'd100 + r * 4 + c + 1);
            end
        end

        repeat (2) @(posedge clk);
        rst_n = 1'b1;

        for (int r = 0; r < 4; r++) begin
            for (int c = 0; c < 4; c++) begin
                write_element(1'b0, r[1:0], c[1:0], a_mat[r][c]);
                write_element(1'b1, r[1:0], c[1:0], b_mat[r][c]);
            end
        end

        feed_en = 1'b1;
        for (int t = 0; t < 7; t++) begin
            cycle_count = t[2:0];
            #1;
            if (a_vec !== exp_a_vec) begin
                $error(
                    "A mismatch at t=%0d observed=%h expected=%h",
                    t,
                    a_vec,
                    exp_a_vec
                );
            end
            if (b_vec !== exp_b_vec) begin
                $error(
                    "B mismatch at t=%0d observed=%h expected=%h",
                    t,
                    b_vec,
                    exp_b_vec
                );
            end
        end

        feed_en = 1'b0;
        cycle_count = 3'd0;
        #1;
        for (int i = 0; i < 4; i++) begin
            if (a_data[i] !== 8'd0 || b_data[i] !== 8'd0) begin
                $error("feed_en low should zero outputs");
            end
        end

        $display("buffer_input_tb passed");
        $finish;
    end

endmodule
