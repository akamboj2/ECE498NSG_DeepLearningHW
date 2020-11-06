//3 tap linear predictor code

module predictor(
	input logic reset, Clk,
	input logic [8:0] xn3, w0, w1, w2,
	output logic [8:0] x_hat, err);

//internal signals across registers
logic [8:0] xn2, xn1, xn;
logic ld;
assign ld = 1; //we are loading the registers every clock cycle
//assign xn3 = x;

//internal signals, outputs between adders and multipliers
logic [8:0] mx0_out, mx1_out, mx2_out, ax01_out, ax2_out, aerr_out, aw0_out, aw1_out, aw2_out, mw0_out, mw1_out, mw2_out;

//internal registers:
register rx1(.data_in(xn3), .data_out(xn2), .*); //.* should connect ld, Clk, reset
register rx2(.data_in(xn2), .data_out(xn1), .*);
register rx3(.data_in(xn1), .data_out(xn), .*);


//filter block:
multiply mx0(.a(xn3), .b(w0), .result(mx0_out));
multiply mx1(.a(xn2), .b(w1), .result(mx1_out));
add ax01(.a(mx0_out), .b(mx1_out), .result(ax01_out));
multiply mx2(.a(xn1), .b(w2), .result(mx2_out));
add ax2(.a(ax01_out), .b(mx2_out), .result(x_hat));

//error
add aerr(.a(~x_hat + 1), .b(xn), .result(err));


//Weight update block:
/*
register rw0(.data_in(aw0_out), .data_out(w0), .*);
register rw1(.data_in(aw1_out), .data_out(w1), .*);
register rw2(.data_in(aw2_out), .data_out(w2), .*);
*/
/*
multiply mw0(.a(err), .b(xn3), .result(mw0_out));
multiply mw1(.a(err), .b(xn2), .result(mw1_out));
multiply mw2(.a(err), .b(xn1), .result(mw2_out));

add aw0(.a(mw0_out), .b(w0), .result(aw0_out));
add aw1(.a(mw1_out), .b(w1), .result(aw1_out));
add aw2(.a(mw2_out), .b(w2), .result(aw2_out));
*/
endmodule

module add(
	input logic [8:0] a, b,
	output logic[8:0] result);
assign result = a + b;//(real'(a) + real'(b));
endmodule

module multiply(
	input logic [8:0] a,b,
	output logic [8:0] result);

assign result = a*b;//(real'(a) * real'(b));

endmodule

module register(
	input logic[8:0] data_in,
	input logic ld, reset, Clk,
	output logic[8:0] data_out);
	
always_ff @(posedge Clk)
begin
	if (reset)
		data_out <= 9'h00;
	else if(ld==1'b1)
		data_out <= data_in;
	else
		data_out <= data_out;
		
end
endmodule 
