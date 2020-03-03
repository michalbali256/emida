
namespace emida
{

template<typename T>
void subtract_mean(T* pic, size_t one_size, size_t b_size)
{
	for (size_t batch = 0; batch < b_size; ++batch)
	{
		T sum = 0;
		for (size_t i = 0; i < one_size; ++i)
			sum += pic[i];

		T avg = sum / one_size;

		for (size_t i = 0; i < one_size; ++i)
			pic[i] -= avg;

		pic += one_size;
	}
}

}