#=============================================================================
# Synthesis Script -- Single-Head Attention Top (attention_top)
# Top module : attention_top
# Source     : ../rtl_top/
# Usage      : dc_shell -f syn.tcl
#=============================================================================

set TOP    "attention_top"
set CONFIG "compile"

# 1. Analyze (leaf modules first, then top)
analyze -format sverilog ../rtl_top/pe.sv
analyze -format sverilog ../rtl_top/systolic_array_8x8.sv
analyze -format sverilog ../rtl_top/softmax_unit.sv
analyze -format sverilog ../rtl_top/buffer_output.sv
analyze -format sverilog ../rtl_top/attention_fsm.sv
analyze -format sverilog ../rtl_top/attention_top.sv

# 2. Elaborate + set current + link
elaborate      $TOP
current_design $TOP
link

# 3. Constraints
source ./attention_top_syn.sdc

# 4. Synthesize
compile_ultra

# 5. Reports + netlist
report_timing  > ${CONFIG}_timing.rpt
report_area    > ${CONFIG}_area.rpt
report_power   > ${CONFIG}_power.rpt
write -format verilog -hierarchy -output ${CONFIG}_${TOP}_netlist.v

puts "Done. Files prefixed with: ${CONFIG}_"
