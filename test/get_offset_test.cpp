#include <gtest/gtest.h>

#include "get_offset.hpp"
#include "matrix.hpp"
#include "double_compare.hpp"
#include "slice_picture.hpp"

using namespace emida;

TEST(slice_picture, small)
{
	std::vector<int> v =
	{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16
	};

	std::vector<int> expected =
	{
		1, 2, 
		5, 6,

		3, 4,
		7, 8,

		9, 10, 
		13, 14,

		11, 12,
		15, 16
	};

	auto sliced = slice_picture(v.data(), { 4, 4 }, { 2, 2 }, {2, 2});

	EXPECT_EQ(sliced, expected);

}

TEST(get_offset, bigger)
{
	matrix<double> pic = matrix<double>::from_file("test/res/data_pic.txt");
	matrix<double> temp = matrix<double>::from_file("test/res/data_temp.txt");

	vec2<size_t> src_size{ pic.n, pic.n };
	vec2<size_t> size{ 64, 64 };

	auto a = get_submatrix<double, double>(pic.data.data(), src_size, { 0, 0 }, size);
	auto b = get_submatrix<double, double>(temp.data.data(), src_size, { 0, 0 }, size);

	auto offset = get_offset<double>(a.data(), b.data(), size, 1);

	//results from test.py, first left topmost square
	//precision 1e-14 is OK, 1e-15 is failing
	EXPECT_NEAR(offset[0].x, 0.07583538046549165, 2e-14);
	EXPECT_NEAR(offset[0].y, -0.0982055210473689, 2e-14);
}

class get_offset_fixture : public ::testing::TestWithParam<std::tuple<size2_t, cross_policy>> {

};


TEST_P(get_offset_fixture, size_)
{
	size2_t cross_size = std::get<0>(GetParam());
	cross_policy cr_policy = std::get<1>(GetParam());

	matrix<double> pic = matrix<double>::from_file("test/res/data_pic.txt");
	matrix<double> temp = matrix<double>::from_file("test/res/data_temp.txt");

	vec2<size_t> src_size{ pic.n, pic.n };
	vec2<size_t> size{ 64, 64 };

	auto a = get_submatrix<double, double>(pic.data.data(), src_size, { 0, 0 }, size);
	auto b = get_submatrix<double, double>(temp.data.data(), src_size, { 0, 0 }, size);

	auto offset = get_offset<double>(a.data(), b.data(), size, cross_size, cr_policy);

	//results from test.py, first left topmost square
	//precision 1e-14 is OK, 1e-15 is failing
	EXPECT_NEAR(offset[0].x, 0.07583538046549165, 2e-14);
	EXPECT_NEAR(offset[0].y, -0.0982055210473689, 2e-14);
}

struct stringer_policy
{
	std::string operator()(::testing::TestParamInfo<cross_policy> p)
	{
		if (p.param == CROSS_POLICY_FFT)
			return "FFT";
		else
			return "BRUTE";
	}
};

struct stringer
{
	std::string operator()(::testing::TestParamInfo<std::tuple<size2_t, cross_policy>> p)
	{
		size2_t cross_size = std::get<0>(p.param);
		cross_policy cr_policy = std::get<1>(p.param);
		std::stringstream ss;
		ss << stringer_policy()(::testing::TestParamInfo<cross_policy>(cr_policy, 0)) << "_" << cross_size.x << "x" << cross_size.y;
		return ss.str();
	}
};

INSTANTIATE_TEST_SUITE_P(
	get_offset,
	get_offset_fixture,
	::testing::Values(
		std::make_tuple <size2_t, cross_policy>({ 3, 3 }, CROSS_POLICY_BRUTE),
		std::make_tuple <size2_t, cross_policy>({ 5, 5 }, CROSS_POLICY_BRUTE),
		std::make_tuple <size2_t, cross_policy>({ 3, 3 }, CROSS_POLICY_FFT)
	),
	stringer()
);



class get_offset_batched : public ::testing::TestWithParam<cross_policy> {

};

TEST_P(get_offset_batched, batched)
{
	cross_policy c_policy = GetParam();

	matrix<double> pic = matrix<double>::from_file("test/res/data_pic.txt");
	matrix<double> temp = matrix<double>::from_file("test/res/data_temp.txt");

	vec2<size_t> src_size{ pic.n, pic.n };
	vec2<size_t> size{ 64, 64 };
	vec2<size_t> step{ 32, 32 };
	size_t batch_size = get_sliced_batch_size(src_size, size, step);

	auto begins = get_slice_begins(src_size, size, step);

	gpu_offset<double, double> offs(src_size, &begins, size, { 127,127 }, c_policy);
	offs.allocate_memory(temp.data.data());
	auto [offsets, _] = offs.get_offset(pic.data.data());
	

	//precision 1e-13 is OK, 1e-14 is failing

	std::vector<vec2<double>> expected =
	{
		{0.07583538046549165, -0.0982055210473689},
		{0.0663767374671167, -0.04069601894434527},
		{0.074197482004692, -0.030619500269573052},
		{0.05436786376404257, -0.005589154658409257},
		{0.06733198717111577, 0.0517236573394797},
		{0.041554775305378655, 0.06618103013488508},
		{0.08077034898616375, 0.06563583626893177},
		{0.02731542373235385, -0.07039470582336094},
		{0.06008171647840754, -0.02744745873626897},
		{0.029170141077287326, -0.023289975910856242},
		{0.0326187646798104, -0.005564197682964789},
		{0.03391155180750616, 0.019564499299086435},
		{0.04385983944938232, 0.05449764848280836},
		{0.02024519655957846, 0.06849415625676158},
		{0.01368431240032919, -0.07111342587600689},
		{0.012048011945815063, -0.03926198717228857},
		{0.013961762064383265, -0.015440254382809826},
		{0.015755335565160067, -0.005555724262663375},
		{0.023421560567179256, 0.009032311534404869},
		{0.023248682942536902, 0.0350821811421298},
		{0.03053119272782112, 0.06845563900186846},
		{-0.0032150199066265372, -0.07778718128383844},
		{-0.005324141702296004, -0.04152366763450033},
		{0.005280972355535596, -0.01804919638136937},
		{-0.0019341378339703397, -0.006479061775870321},
		{0.0008047285811727534, 0.017339600500093866},
		{-0.003395056024281473, 0.05269941701205738},
		{-0.011578085777955494, 0.05729089241229701},
		{-0.005359767640584323, -0.05472352384999368},
		{-0.05244171495544947, -0.04443498940228352},
		{-0.01161102567282768, -0.017353308868052864},
		{-0.008499046745086503, -0.004588709518344558},
		{-0.01571490670305309, 0.007374566094114243},
		{-0.007446791064488423, 0.039194860149095234},
		{-0.036628348106169994, 0.08157786777366738},
		{-0.023866359887549038, -0.051201391631913395},
		{-0.046419482310795956, -0.06510193845873857},
		{-0.047653378663540025, -0.03436734987603529},
		{-0.05175330219534402, -0.008762911732411283},
		{-0.05304557884009853, 0.02318941866170121},
		{-0.05468521432916873, 0.07303630737680322},
		{-0.0748218633099853, 0.08867438132940464},
		{-0.05947110619956675, -0.058893164709672874},
		{-0.04432777844017011, -0.04082147832301786},
		{-0.05327469916555572, -0.03227153665071825},
		{-0.06626366960730223, 0.013074685592769697},
		{-0.06791710237803272, 0.040805467203661294},
		{-0.047683181620783444, 0.03318911716604589},
		{-0.11304959759046085, 0.061363430916259176},
	};


	EXPECT_VEC_VECTORS_NEAR(offsets, expected, 7e-14);
}

INSTANTIATE_TEST_SUITE_P(
	get_offset,
	get_offset_batched,
	::testing::Values(
		CROSS_POLICY_BRUTE, CROSS_POLICY_FFT),
	stringer_policy()
);


#ifdef MEASURE_TIME

void bbb(int repeat)
{
	std::vector<vec2<double>> expected =
	{
		{0.07583538046549165, -0.0982055210473689},
		{0.0663767374671167, -0.04069601894434527},
		{0.074197482004692, -0.030619500269573052},
		{0.05436786376404257, -0.005589154658409257},
		{0.06733198717111577, 0.0517236573394797},
		{0.041554775305378655, 0.06618103013488508},
		{0.08077034898616375, 0.06563583626893177},
		{0.02731542373235385, -0.07039470582336094},
		{0.06008171647840754, -0.02744745873626897},
		{0.029170141077287326, -0.023289975910856242},
		{0.0326187646798104, -0.005564197682964789},
		{0.03391155180750616, 0.019564499299086435},
		{0.04385983944938232, 0.05449764848280836},
		{0.02024519655957846, 0.06849415625676158},
		{0.01368431240032919, -0.07111342587600689},
		{0.012048011945815063, -0.03926198717228857},
		{0.013961762064383265, -0.015440254382809826},
		{0.015755335565160067, -0.005555724262663375},
		{0.023421560567179256, 0.009032311534404869},
		{0.023248682942536902, 0.0350821811421298},
		{0.03053119272782112, 0.06845563900186846},
		{-0.0032150199066265372, -0.07778718128383844},
		{-0.005324141702296004, -0.04152366763450033},
		{0.005280972355535596, -0.01804919638136937},
		{-0.0019341378339703397, -0.006479061775870321},
		{0.0008047285811727534, 0.017339600500093866},
		{-0.003395056024281473, 0.05269941701205738},
		{-0.011578085777955494, 0.05729089241229701},
		{-0.005359767640584323, -0.05472352384999368},
		{-0.05244171495544947, -0.04443498940228352},
		{-0.01161102567282768, -0.017353308868052864},
		{-0.008499046745086503, -0.004588709518344558},
		{-0.01571490670305309, 0.007374566094114243},
		{-0.007446791064488423, 0.039194860149095234},
		{-0.036628348106169994, 0.08157786777366738},
		{-0.023866359887549038, -0.051201391631913395},
		{-0.046419482310795956, -0.06510193845873857},
		{-0.047653378663540025, -0.03436734987603529},
		{-0.05175330219534402, -0.008762911732411283},
		{-0.05304557884009853, 0.02318941866170121},
		{-0.05468521432916873, 0.07303630737680322},
		{-0.0748218633099853, 0.08867438132940464},
		{-0.05947110619956675, -0.058893164709672874},
		{-0.04432777844017011, -0.04082147832301786},
		{-0.05327469916555572, -0.03227153665071825},
		{-0.06626366960730223, 0.013074685592769697},
		{-0.06791710237803272, 0.040805467203661294},
		{-0.047683181620783444, 0.03318911716604589},
		{-0.11304959759046085, 0.061363430916259176},
	};

	//START_STOPWATCH()
	matrix<double> pic = matrix<double>::from_file("test/res/data_pic.txt");
	matrix<double> temp = matrix<double>::from_file("test/res/data_temp.txt"); //TICK("From file: ");

	vec2<size_t> src_size{ pic.n, pic.n };
	vec2<size_t> size{ 64, 64 };
	vec2<size_t> step{ 32, 32 };
	size_t batch_size = get_sliced_batch_size(src_size, size, step);

	auto a = slice_picture(pic.data.data(), src_size, size, step);
	auto b = slice_picture(temp.data.data(), src_size, size, step); //TICK("Slice: ");
	
	
	a = repeat_vector(a, repeat);
	b = repeat_vector(b, repeat);
	expected = repeat_vector(expected, repeat);
	std::cout << a.size() * sizeof(double) << "B\n";
	auto offset = get_offset<double>(a.data(), b.data(), size, {3,3}, batch_size * repeat);
	//TOTAL();

	EXPECT_VEC_VECTORS_NEAR(offset, expected, 7e-14);
}

TEST(get_offset, time10)
{
	bbb(10);
}

TEST(get_offset, time100)
{
	bbb(100);
}

TEST(get_offset, time200)
{
	bbb(200);
}

TEST(get_offset, time300)
{
	bbb(300);
}

#endif 
