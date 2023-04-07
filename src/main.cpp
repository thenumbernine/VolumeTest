#include "Common/Macros.h"		//crand
#include "Common/Sequence.h"	//make_index_range
#include "Tensor/Tensor.h"
#include <sys/time.h>
#include <stdlib.h>
#include <iostream>

using real = double;

constexpr size_t maxdim = 9;	// 10 is too much for release mode.  8 or 9 is too much for debug mode.

template<int i, int dim>
struct WedgeNTimes {
	static constexpr auto go(auto const & v) {
		return v[i].wedge(
			WedgeNTimes<i+1, dim>::go(v)
		);
	}
};
template<int dim>
struct WedgeNTimes<dim-1, dim> {
	static constexpr auto go(auto const & v) {
		return v[dim-1];
	}
};


// sdim = simplex-dimension.  
// for the first case it will match ws.dim<0> == ws.dim<1> (since ws.rank == 2)
// for the rest, ws.dim<0> > sdim, so we gotta pass sdim
template<size_t sdim>	
void testVolume(auto const & ws) {
	auto w = WedgeNTimes<0, sdim>::go(ws);
//	std::cout << "w = " << sdim << "-simplex in " << sdim << " dimensions:" << std::endl;
	//auto wexp = w.expand();
	//std::cout << wexp << std::endl;
//	std::cout << "*w: " << w.dual() << std::endl;
	//std::cout << "*w expanded (should match): " << wexp.dual() << std::endl;
	auto wInner = w.wedge(w.dual()).dual();
//	std::cout << "*(w ∧ *w): " << wInner << std::endl;
	std::cout << "√(*(w ∧ *w)): " << sqrt(wInner) << std::endl;
	// frob will match inner in rank-1 norms only
//	auto wFrob = wexp.lenSq();
//	std::cout << "|w|_frob: " << wFrob << std::endl;	// Frobenius norm
//	std::cout << "√(|w|_frob): " << sqrt(wFrob) << std::endl;	// Frobenius norm
}

#if 0
template<
	size_t sdim,
	size_t dim
>
void testSimplexDimForDim() {
	std::cout << "simplex dim " << sdim << " dim " << dim << std::endl;

	//'dim' is the dim of the vectors
	//'sdim' is the simplex dim <-> how many vectors
	//Tensor::vec<real, dim>
}
#else
template<int x>
struct Inc {
	static constexpr int value = x + 1;
};

template<
	size_t sdim,	//simplex dimension
	size_t dim,		//dimension to test in
	size_t stopdim	//dimension to stop at
>
struct TestSimplexDimForDim {
	template<typename W>
	static constexpr void go(W const & ws) {
		std::cout << " === rotating from dim " << (dim-1) << " into dim " << dim << std::endl;
		// start with rank-n dim-n
		// then rotate each component into dim-(n+1)
		static_assert(W::rank == 2);
		static_assert(W::template dim<0> == dim-1);
		static_assert(W::template dim<1> == dim-1);
		//std::cout << "ws dim " << W::dims() << std::endl;


#if 0 // TODO how to use fold/seq to increment a seq by 1...
		auto nws = []<auto ... i>(std::index_sequence<i...>) constexpr {
			return Tensor::tensor<real, (W::template dim<i>)... + 1>();
		}(std::make_index_sequence<W::rank>{});
#else	//until then
		
		// TODO instead of SeqToSeqMap using ::value, have it use operator()
		using NW = Tensor::tensorScalarSeq<
			real,
			Common::SeqToSeqMap<typename W::dimseq, Inc>
		>;
		
		auto nws = NW(ws);
		nws(dim-1, dim-1) = 1;
		//nws(int2{dim-1,dim-1}) = 1;
//		std::cout << "ws: " << ws << std::endl;
//		std::cout << "nws before rotate: " << nws << std::endl;

#if 1	//rotate into the new dimension
		//ok now rotate legs ... 
		for (size_t i = 0; i < dim-1; ++i) {
			size_t j = dim-1; {
			//for (size_t j = i + 1; j < dim; ++j) {
				auto rot = NW([](int i, int j) -> real { return i == j ? 1 : 0; });
				real theta = frand() * 2. * M_PI;
				real costh = cos(theta);
				real sinth = sin(theta);
				rot(i,i) = rot(j,j) = costh;
				rot(i,j) = -(rot(j,i) = sinth);
#if 0
				constexpr auto a = Tensor::Index<'a'>{};
				constexpr auto b = Tensor::Index<'b'>{};
				constexpr auto c = Tensor::Index<'c'>{};
				nws(a,b) = nws(a,c) * rot(c,b);
#else
				// each row is a distinct base vector of our simplex
				// so just rotate the row components
				//nws[i] = the i'th vector, rotate each i'th vector's components, so nws[i][k] * rot[k][j]
				nws = nws * rot;
#endif
			}
		}
#endif
#endif
		
//		std::cout << "nws after rotate: " << nws << std::endl;

		testVolume<sdim>(nws);

		TestSimplexDimForDim<sdim, dim+1, stopdim>::template go<NW>(nws);
	}
};
template<
	size_t sdim,
	size_t stopdim
>
struct TestSimplexDimForDim<sdim, stopdim, stopdim> {
	static constexpr void go(auto const & v) {}
};
#endif

template<size_t dim>
auto randomVec() {
	auto w = Tensor::vec<real, dim>([](int i) { return 10. * crand(); });
	std::cout << "generating " << dim << "-dimensional random vector: " << w << std::endl;
	return w;
}

template<size_t sdim>
void testSimplexDim() {
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

	testVolume<sdim>(ws);

#if 0 // would be nice but I wanna use the prev function in the next, soo... another case of fold to repeatly apply a function (how to?)
	[]<auto ... j>(std::index_sequence<j...>) constexpr {
		(testSimplexDimForDim<sdim,j>(w), ...);
	}(Common::make_index_range<sdim+1, maxdim+1>{});
#else
	TestSimplexDimForDim<sdim, sdim+1, maxdim+1>::go(ws);
#endif
}

int main() {
	srand(time(nullptr));
	// for constexpr (size_t i = 1; i <= maxdim; ++i) {
	[]<auto ... i>(std::index_sequence<i...>) constexpr {
		(testSimplexDim<i>(), ...);
	}(Common::make_index_range<1, maxdim+1>{});
}
