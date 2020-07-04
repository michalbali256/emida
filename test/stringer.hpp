#include <gtest/gtest.h>

template<typename T>
struct stringer
{
	std::string operator()(::testing::TestParamInfo<T> p)
	{
		return p.param.name;
	}
};