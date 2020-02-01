#pragma once

#include <fstream>
#include <vector>
namespace emida
{
template<typename T>
class matrix
{
public:
	size_t n;
	std::vector<T> data;

	static matrix<T> from_file(std::string file_name)
	{
		matrix<T> m;
		std::ifstream f(file_name);
		if (!f.good())
			throw std::exception(("File " + file_name + " cannot be opened.").c_str());
		f >> m.n;
		m.data.resize(m.n * m.n);
		for (size_t i = 0; i < m.n * m.n; ++i)
			f >> m.data[i];
		f.close();
		return m;
	}

	bool operator==(const matrix& rhs) const
	{
		return data == rhs.data;
	}
};

}