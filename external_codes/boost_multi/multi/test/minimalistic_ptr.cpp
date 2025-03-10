// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
// © Alfredo A. Correa 2018-2021

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi minimalistic pointer"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include<array>

#include "../array_ref.hpp"

namespace multi = boost::multi;

namespace minimalistic {

template<class T> class ptr : public std::iterator_traits<T*>{ // minimalistic pointer
	using underlying_type = T*;
	underlying_type impl_;
	template<class> friend class ptr;

 public:
	ptr() = default;
	constexpr explicit ptr(T* impl) : impl_{impl} {}
	template<class U, class = std::enable_if_t<std::is_convertible<U*, T*>{}> >
	// cppcheck-suppress noExplicitConstructor
	ptr(ptr<U> const& other) : impl_{other.impl_} {} //  NOLINT(google-explicit-constructor, hicpp-explicit-conversions): ptr<T> -> ptr<T const>
	using typename std::iterator_traits<T*>::reference;
	using typename std::iterator_traits<T*>::difference_type;
	// NOLINTNEXTLINE(fuchsia-overloaded-operator, fuchsia-trailing-return): operator* used because this class simulates a pointer, trailing return helps
	constexpr auto operator*() const -> reference {return *impl_;}
	// NOLINTNEXTLINE(fuchsia-overloaded-operator, cppcoreguidelines-pro-bounds-pointer-arithmetic): operator+ is overloaded to simulate a pointer
	constexpr auto operator+(difference_type n) const {return ptr{impl_ + n};}
//	T& operator[](difference_type n) const{return impl_[n];} // optional
	using default_allocator_type = std::allocator<T>;
	template<class> friend class ptr2;
};

template<class T> class ptr2 : public std::iterator_traits<T*>{ // minimalistic pointer
	T* impl_;
 public:
	constexpr explicit ptr2(T* impl) : impl_{impl} {}
	constexpr explicit ptr2(ptr<T> const& p) : impl_{p.impl_} {}
	template<class U, class = std::enable_if_t<std::is_convertible<U*, T*>{}> >
	// cppcheck-suppress noExplicitConstructor
	ptr2(ptr2<U> const& other) : impl_{other.impl_} {} // NOLINT(google-explicit-constructor, hicpp-explicit-conversions): ptr<T> -> ptr<T const>
	using typename std::iterator_traits<T*>::reference;
	using typename std::iterator_traits<T*>::difference_type;
	// NOLINTNEXTLINE(fuchsia-overloaded-operator, fuchsia-trailing-return): operator* used because this class simulates a pointer, trailing return helps
	constexpr auto operator*() const -> reference {return *impl_;}
	// NOLINTNEXTLINE(fuchsia-overloaded-operator, cppcoreguidelines-pro-bounds-pointer-arithmetic): operator+ is overloaded to simulate a pointer
	constexpr auto operator+(typename ptr2::difference_type n) const {return ptr2{impl_ + n};}
//	T& operator[](std::ptrdiff_t n) const{return impl_[n];} // optional
	using default_allocator_type = std::allocator<T>;
};

} // end namespace minimalistic

BOOST_AUTO_TEST_CASE(test_minimalistic_ptr) {
	std::array<double, 400> buffer{};

	using pointer_type = minimalistic::ptr<double>;
	multi::array_ptr<double, 2, pointer_type> CCP(pointer_type{buffer.data()}, {20, 20});
	(*CCP)[2]; // requires operator+
	(*CCP)[1][1]; // requires operator*
	(*CCP)[1][1] = 9;
	BOOST_REQUIRE( &(*CCP)[1][1] == &buffer[21] );

	auto&& CC2 = CCP->static_array_cast<double, minimalistic::ptr2<double>>();
	BOOST_REQUIRE( &CC2[1][1] == &(*CCP)[1][1] );

	static_assert( std::is_convertible<double*, double const*>{}, "!");

	minimalistic::ptr<double> pd{nullptr};
	minimalistic::ptr<const double> pcd = pd;
	(void)pcd;

	{
		auto&& REF = *CCP; (void)REF;
		static_assert( std::is_same<decltype(REF.partitioned(2).partitioned(2).base()), minimalistic::ptr<double>>{}, "!" );
	}
	 {
		auto const& REF = *CCP; (void)REF;
		static_assert( std::is_same<decltype(REF.partitioned(2).partitioned(2).base()), minimalistic::ptr<double const>>{}, "!" );
	}
}



