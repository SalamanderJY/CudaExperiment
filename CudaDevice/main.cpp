
// print device parameters of cuda.
extern int parameters_main(int argc, char* argv[]);

extern int vectorsum_main(int argc, char* argv[]);

int main(int argc, char* argv[])
{
	return parameters_main(argc, argv);
	//return vectorsum_main(argc, argv);

}