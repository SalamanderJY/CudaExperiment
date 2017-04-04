// cuda auto generate function with vector demo.
extern int demo_main(int argc, char* argv[]);

// print device parameters of cuda.
extern int parameters_main(int argc, char* argv[]);

extern int vectorsum_main(int argc, char* argv[]);

extern int warp_main(int argc, char* argv[]);

extern int atomic_main(int argc, char* argv[]);

int main(int argc, char* argv[])
{
	//return demo_main(argc, argv);
	//return parameters_main(argc, argv);
	//return vectorsum_main(argc, argv);
	return atomic_main(argc, argv);

}