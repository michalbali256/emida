
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

#define EXPECT_VEC_VECTORS_EQ(a, b) \
	ASSERT_EQ(a.size(), b.size()) << "Vectors x and y are of unequal length"; \
	for (int i = 0; i < a.size(); ++i) \
	{ \
		EXPECT_DOUBLE_EQ(a[i].x, b[i].x) << "Vectors " #a " and " #b " differ at x at index " << i; \
		EXPECT_DOUBLE_EQ(a[i].y, b[i].y) << "Vectors " #a " and " #b " differ at y at index " << i; \
	}
