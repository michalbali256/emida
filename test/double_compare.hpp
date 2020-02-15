
#define EXPECT_DOUBLE_VECTORS_EQ(x,y) \
	ASSERT_EQ(x.size(), y.size()) << "Vectors x and y are of unequal length"; \
	for (int i = 0; i < x.size(); ++i) \
	{ \
		EXPECT_DOUBLE_EQ(x[i], y[i]) << "Vectors " #x " and " #y " differ at index " << i; \
	}

#define EXPECT_DOUBLE_VECTORS_NEAR(x,y, abs_error) \
	ASSERT_EQ(x.size(), y.size()) << "Vectors x and y are of unequal length"; \
	for (int i = 0; i < x.size(); ++i) \
	{ \
		EXPECT_NEAR(x[i], y[i], abs_error) << "Vectors " #x " and " #y " differ at index " << i; \
	}