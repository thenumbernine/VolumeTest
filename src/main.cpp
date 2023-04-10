#include "Common/Macros.h"		//crand
#include "Common/Sequence.h"	//make_index_range
#include "Tensor/Tensor.h"
#include <sys/time.h>
#include <stdlib.h>
#include <iostream>
#include <cassert>

#if 0
#include "Profiler/Profiler.h"
#else
#define PROFILE_BEGIN_FRAME()
#define PROFILE_END_FRAME()
#define PROFILE()
#endif

#include <sstream>

namespace Tensor {

// TODO move this to Tensor?  and clean it up maybe?
template<typename N, int i>
void serializeIndex(auto & o) {
	using Ni = std::remove_pointer_t<std::tuple_element_t<i, N>>;
	o << " " << Ni::tensorxStr();
};
template<typename T>
std::string tensorxName(T const & t) {
	using N = typename T::NestedPtrTensorTuple;
	std::ostringstream o;
	[]<auto ... i>(auto & o, std::index_sequence<i...>) constexpr {
		(serializeIndex<N,i>(o), ...);
	}(o, std::make_index_sequence<std::tuple_size_v<N>>{});
	return o.str();
}

template<int b>
constexpr int ipow(int a) {
	if constexpr (b == 0) {
		return 1;
	} else if constexpr (b == 1) {
		return a;
	} else {
		return a * ipow<b-1>(a);
	}
}

//inner product is going slow
// because read iterator is going slow
template<typename A, typename B>
requires (
	IsBinaryTensorOp<A,B> || 				// ... with matching rank
	(!is_tensor_v<A> && !is_tensor_v<B>)	// ... or two scalars
)
auto innerProfile(A const & a, B const & b) {
	PROFILE();
	if constexpr (!is_tensor_v<A> || !is_tensor_v<B>) {	// two scalars:
		PROFILE();
		return a * b;
	} else {
		PROFILE();
		using AS = typename A::Scalar;
		using BS = typename B::Scalar;
		using RS = decltype(AS() * BS());
//std::cout << typeid(A).name() <<  << std::endl;
//how to serialize the tensorx type?
// NestedPtrTensorTuple
std::cout << tensorxName(a) << " inner " << tensorxName(b) << std::endl;
//std::cout << "sized " << a.localCount << " x " << b.localCount << std::endl;
		/*
		vec
		ident
		zero
		sym
		asym
		symR
		asymR
		
		how can I optimize this?
		if A is sym and B is sym (or A is asym and B is asym) then we can double up the symmetric indexes (same with symR) ... but how many times?
		*/
	
		if constexpr (is_zero_v<A> || is_zero_v<B>) {
		//if A or B is a zero then return zero.
			PROFILE();
			return RS{};
		} else if constexpr (is_ident_v<A> || is_ident_v<B>) {
		//if A is an ident or B is an ident then return the trace of inners of a and b
			PROFILE();
			RS sum = {};
			for (int i = 0; i < a.localDim; ++i) {
				sum += innerProfile(a(i,i), b(i,i));
			}
			return sum;
		} else if constexpr (
			std::is_same_v<A, B>	// TODO test if matching up to scalar types
			&& (is_asym_v<A> || is_asymR_v<A>)	// ... then ... we can short circuit ...
			&& sizeof(A) == sizeof(typename A::Scalar)	// TODO how about optimizing for all cases?
			// this would involve iterate/sum-of-products over all localCount storage
			// with products weighted by some factor
		) {
			PROFILE();
			// TODO wrong
			// inner(a,b) produces the correct result
			//return inner(a,b);
			// a.s[0] * b.s[0] produces the incorrect result
			return innerProfile(a.s[0], b.s[0]) * (RS)constexpr_factorial(A::localRank);
				// * (RS)ipow<A::localRank>(A::localDim);
				// * (RS)consteval_nChooseR(A::localRank, A::localDim);
#if 0
		// if *any* neighboring indexes is of a sym(R) in A and asym(R) in B (or vice versa) then the result is zero (same with symR)
		// i.e. a_i1_..._[ik_i{k+1}] b^i1^...^(ik^i{k+1}) = 0
		// i.e. for rank-k, iterator i=0..k-2, 
		//		if the nesting-index in A of i == nesting-index of i+1 
		//	... and same with nesting-index in B of i == nesting-index of i+1
		//	... and is_sym(R) of that nesting in A and is_asym(R) of that nesting in B (or vice versa)
		//	... then return 0
		} else if constexpr (
			false
		) {
#endif	
		} else {
		//otherwise old fashioned
			PROFILE();
			//ok asymR inner asymR is going to iterate over its same component many times ...
			auto i = a.begin();
			auto sum = a(i.index) * b(i.index);
			for (++i; i != a.end(); ++i) {
				sum += a(i.index) * b(i.index);
			}
			return sum;
		}
	}
}

template<int num, typename A, typename B>
requires IsInteriorOp<num, A, B>
auto interiorProfile(A const & a, B const & b) {
#if 0
	return contractN<A::rank-num,num>(outer(a,b));
#else
	PROFILE();
	using S = typename A::Scalar;
	if constexpr (A::rank == num && B::rank == num) {
		PROFILE();
		// rank-0 i.e. scalar result case
		static_assert(std::is_same_v<typename A::dimseq, typename B::dimseq>);	//thanks to the 3rd requires condition
		return innerProfile(a,b);
	} else {
		PROFILE();
		using R = typename A
			::template ReplaceScalar<B>
			::template RemoveIndexSeq<Common::make_integer_range<int, A::rank-num, A::rank+num>>;
		static_assert(num != 1 || std::is_same_v<R, decltype(contract<A::rank-1,A::rank>(outer(a,b)))>);
		static_assert(std::is_same_v<R, decltype(contractN<A::rank-num,num>(outer(a,b)))>);
		static_assert(R::rank == A::rank + B::rank - 2 * num);
		return R([&](typename R::intN i) -> S {
			PROFILE();	// #3 offender
			auto ai = [&]<int ... j>(std::integer_sequence<int, j...>) constexpr -> typename A::intN {
				//PROFILE();	//clang can't put PROFILE in constexpr lambdas
				return typename A::intN{(j < A::rank-num ? i[j] : 0)...};
			}(std::make_integer_sequence<int, A::rank>{});
			auto bi = [&]<int ... j>(std::integer_sequence<int, j...>) constexpr -> typename B::intN {
				//PROFILE();	//clang can't put PROFILE in constexpr lambdas
				return typename B::intN{(j < num ? 0 : i[j + A::rank-2*num])...};
			}(std::make_integer_sequence<int, B::rank>{});

			//TODO instead use A::dim<A::rank-num..A::rank>
			S sum = {};
#if 0
			template<typename B>
			struct InteriorRangeIter {
				template<int i> constexpr int getRangeMin() const { return 0; }
				template<int i> constexpr int getRangeMax() const { return B::dims().template dim<i>; }
			};
			for (auto k : RangeIteratorInner<num, InteriorRangeIter<B>>(InteriorRangeIter<B>())) {
#else
			for (auto k : RangeObj<num, false>(vec<int, num>(), B::dims().template subset<num, 0>())) {
				PROFILE();	// #2 offender ... #1 offender with inner optimized for asymR
#endif
				std::copy(k.s.begin(), k.s.end(), ai.s.begin() + (A::rank - num));
				std::copy(k.s.begin(), k.s.end(), bi.s.begin());
				sum += a(ai) * b(bi);
			}
			return sum;
		});
	}
#endif
}

template<typename A>
requires IsSquareTensor<A>
auto hodgeDualProfile(A const & a) {
	PROFILE();
	static constexpr int rank = A::rank;
	static constexpr int dim = A::template dim<0>;
	static_assert(rank <= dim);
	using S = typename A::Scalar;
	if constexpr (dim == 1 && rank == 1) {	// very special case:
		PROFILE();
		return a[0];
	} else if constexpr (dim == 2) {	// TODO this condition isn't needed if you merge asym with asymR
		PROFILE();
		return interiorProfile<rank>(a, asym<S, dim>(1)) / (S)constexpr_factorial(rank);
	} else {
		PROFILE();
		auto e = [&](){
			PROFILE();
			return asymR<S, dim, dim>(1);
		}();
		auto div = [&](){
			PROFILE();
			return (S)constexpr_factorial(rank);
		}();
		auto ae = [&](){
			PROFILE();
			return interiorProfile<rank>(a, e);
		}();
		auto dual = [&](){
			PROFILE();
			return ae / div;
		}();
		return dual;
	}
}

}




using real = double;

constexpr size_t maxdim = 8;	// 10 is too much for release mode.  8 or 9 is too much for debug mode.

// for the first case it will match ws.dim<0> == ws.dim<1> (since ws.rank == 2)
// for the rest, ws.dim<0> < ws.dim<1>
void testVolume(auto const & ws) {
	// ws is tensor (sdim, dim)
	PROFILE();
//	std::cout << "ws: " << ws << std::endl;
	auto w = [&](){
		PROFILE();
		return ws.wedgeAll();
	}();
	// w is tensor(dim) wedge ... wedge tensor(dim) (sdim-times)
	// so w is tensor(dim, dim, ...) totally-antisymmetric, rank<=sdim
	// so it'll go from 'dim' unique elements down to 1 unique element
	// number of unique elements is (dim ncr sdim)
//	constexpr int sdim = std::decay_t<decltype(ws)>::template dim<0>;
//	std::cout << "w = " << sdim << "-simplex in " << sdim << " dimensions:" << std::endl;
	std::cout << "w: " << w << std::endl;	
	auto wDual = [&](){
		PROFILE();
		return Tensor::hodgeDualProfile(w);
	}();
//	auto wexp = w.expand();
//	std::cout << wexp << std::endl;
//	std::cout << "*w: " << wDual << std::endl;
//	std::cout << "*w expanded (should match): " << wexp.dual() << std::endl;
	auto wInnerForm = [&](){
		PROFILE();
		return w.wedge(wDual);
	}();
	auto wInner = [&](){
		PROFILE();
		return Tensor::hodgeDualProfile(wInnerForm);
	}();
//	std::cout << "*(w ∧ *w): " << wInner << std::endl;
	[&](){
		PROFILE();
		std::cout << "√(*(w ∧ *w)): " << sqrt(wInner) << std::endl;
	}();
	// frob will match inner in rank-1 norms only
//	auto wFrob = wexp.lenSq();
//	std::cout << "|w|_frob: " << wFrob << std::endl;	// Frobenius norm
//	std::cout << "√(|w|_frob): " << sqrt(wFrob) << std::endl;	// Frobenius norm
}

template<
	size_t dim,		//dimension to test in
	size_t stopdim	//dimension to stop at
>
void TestSimplexDimForDim(auto const & ws) {
	if constexpr (dim < stopdim) {
		std::cout << " === rotating from dim " << (dim-1) << " into dim " << dim << std::endl;
		// start with rank-n dim-n
		// then rotate each component into dim-(n+1)
		using W = std::decay_t<decltype(ws)>;
		constexpr int sdim = W::template dim<0>;
		static_assert(W::template dim<1> == dim-1);
		//std::cout << "ws dim " << W::dims() << std::endl;


#if 0 // TODO how to use fold/seq to increment a seq by 1...
		auto nws = []<auto ... i>(std::index_sequence<i...>) constexpr {
			return Tensor::tensor<real, (W::template dim<i>)... + 1>();
		}(std::make_index_sequence<W::rank>{});
#else	//until then
		
		// TODO instead of SeqToSeqMap using ::value, have it use operator()
		using NW = Tensor::tensor<real, sdim, dim>;
		auto nws = NW(ws);
		//nws(int2{dim-1,dim-1}) = 1;
//		std::cout << "ws: " << ws << std::endl;
//		std::cout << "nws before rotate: " << nws << std::endl;

		using R = Tensor::tensor<real, dim, dim>;

#if 1	//rotate into the new dimension
		//ok now rotate legs ... 
		for (size_t i = 0; i < dim-1; ++i) {
			size_t j = dim-1; {
			//for (size_t j = i + 1; j < dim; ++j) {
				auto rot = R([](int i, int j) -> real { return i == j ? 1 : 0; });
				real theta = frand() * 2. * M_PI;
				real costh = cos(theta);
				real sinth = sin(theta);
				rot(i,i) = rot(j,j) = costh;
				rot(i,j) = -(rot(j,i) = sinth);
#if 1	// works
				constexpr auto a = Tensor::Index<'a'>{};
				constexpr auto b = Tensor::Index<'b'>{};
				constexpr auto c = Tensor::Index<'c'>{};
				nws(a,b) = nws(a,c) * rot(c,b);
#elif 1	// works
				// each row is a distinct base vector of our simplex
				// so just rotate the row components
				//nws[i] = the i'th vector, rotate each i'th vector's components, so nws[i][k] * rot[k][j]
				nws *= rot;
#endif
			}
		}
#endif
#endif
		
//		std::cout << "nws after rotate: " << nws << std::endl;

		testVolume(nws);

		TestSimplexDimForDim<dim+1, stopdim>(nws);
	}
}

template<size_t dim>
auto randomVec() {
	PROFILE();
	auto w = Tensor::vec<real, dim>([](int i) { return 10. * crand(); });
	std::cout << "generating " << dim << "-dimensional random vector: " << w << std::endl;
	return w;
}

template<size_t sdim>
void testSimplexDim() {
	PROFILE();
	std::cout << std::endl;
	std::cout << "testing simplex of dimension " << sdim << std::endl;

	//start with a random 'sdim'-dimensioned vector
#if 0	//can you use fold expressions to self-apply a function only N times 
	// I guess I can if I overload the custom class' binary operators ...
	auto w = []<auto ... j>(std::index_sequence<j...>) constexpr {
		return randomVec<sdim>().wedge(...);
	}(Common::make_index_sequence<sdim>{});
#else	//old fashioned way:
	//gonna need those base vectors for later for rotating
	// rows are the base vectors
#if 0	// TODO do I not have vec<real,1,1> lambda ctors?
	auto ws = Tensor::tensor<real, sdim, sdim>([]() { return crand() * 10.; });
#else
	auto ws = Tensor::tensor<real, sdim, sdim>{};
	for (size_t i = 0; i < sdim; ++i) {
		for (size_t j = 0; j < sdim; ++j) {
			ws[i][j] = crand() * 10;
		}
	}
#endif
#endif

	testVolume(ws);

#if 0 // would be nice but I wanna use the prev function in the next, soo... another case of fold to repeatly apply a function (how to?)
	[]<auto ... j>(std::index_sequence<j...>) constexpr {
		(testSimplexDimForDim<sdim,j>(w), ...);
	}(Common::make_index_range<sdim+1, maxdim+1>{});
#else
	TestSimplexDimForDim<sdim+1, maxdim+1>(ws);
#endif
}

int main() {
	PROFILE_BEGIN_FRAME()
	PROFILE();
	srand(time(nullptr));
#if 0
	// for constexpr (size_t i = 1; i <= maxdim; ++i) {
	[]<auto ... i>(std::index_sequence<i...>) constexpr {
		(testSimplexDim<i>(), ...);
	}(Common::make_index_range<1, maxdim+1>{});
#else

	// turns out rank 1x1x...xN ctors doesn't' work ... unless it's 1x1x ... x1
#if 1
	// TODO why isn't this working?
	using real2 = Tensor::tensor<real, 2>;
	auto i = real2{1,3};
	auto j = real2{2,4};
	using real1x2 = Tensor::tensor<real, 1, 2>;
#if 1	// works
	auto ii = real1x2{i};
	auto jj = real1x2{j};
#endif
#if 0	// fails
	auto ii = real1x2(i);	// doesn't work
	auto jj = real1x2(j);
#endif
	assert(innerProfile(ii, jj) == 14);
	assert(innerProfile(Tensor::tensor<real, 1, 3>{{1,3,2}}, Tensor::tensor<real, 1, 3>{{2,4,3}}) == 20);
#endif



	// rank-1 vectors
	assert(innerProfile(Tensor::tensor<real, 1>{3}, Tensor::tensor<real, 1>{4}) == 12);
	assert(innerProfile(Tensor::tensor<real, 2>{1,3}, Tensor::tensor<real, 2>{2,4}) == 14);
	assert(innerProfile(Tensor::tensor<real, 3>{1,3,2}, Tensor::tensor<real, 3>{2,4,3}) == 20);

	// rank-2 dense vectors
	assert(innerProfile(Tensor::tensor<real, 1, 1>{{3}}, Tensor::tensor<real, 1, 1>{{5}}) == 15);
	assert(innerProfile(Tensor::tensor<real, 2, 1>{{1},{3}}, Tensor::tensor<real, 2, 1>{{2},{4}}) == 14);

	assert(innerProfile(Tensor::tensor<real, 2, 2>{{1, 2},{3, 4}}, Tensor::tensor<real, 2, 2>{{4, 3},{2, 1}}) == 20);
	
	assert(innerProfile(Tensor::tensor<real, 3, 1>{{1},{3},{2}}, Tensor::tensor<real, 3, 1>{{2},{4},{3}}) == 20);
	
	assert(innerProfile(Tensor::tensor<real, 3, 2>{{1,2},{3,4},{2,5}}, Tensor::tensor<real, 3, 2>{{2,1},{4,2},{3,3}}) == 45);
	assert(innerProfile(Tensor::tensor<real, 3, 3>{{1,2,3},{4,5,6},{7,8,9}}, Tensor::tensor<real, 3, 3>{{9,8,7},{6,5,4},{3,2,1}}) == 165);
	


	// rank-3 dense

	// rank-2 ident

	// rank-2 sym*sym
	// rank-2 sym*asym
	// rank-2 asym*sym
	// rank-2 asym*asym

	// rank-3 same

	// rank-4 same

	// rank-4 ident outer rank-2

#endif
	PROFILE_END_FRAME()
}
