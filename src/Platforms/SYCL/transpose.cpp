#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <numeric>

template<typename T1, typename T2>
sycl::event transpose(sycl::queue& q,
                      const T1* restrict in,
                      int m,
                      int lda,
                      T2* restrict out,
                      int n,
                      int ldb,
                      const std::vector<sycl::event>& events = {})
{ 
  const size_t tile_size = 32;
  const size_t block_rows = 8;
  const size_t n_tiles = ((m + tile_size - 1) / tile_size);

  return q.submit([&](sycl::handler& cgh) {
      cgh.depends_on(events);
      sycl::accessor<T2, 2, sycl::access::mode::write, sycl::access::target::local>
      tile(sycl::range<2>(tile_size,tile_size+1), cgh);
      cgh.parallel_for(sycl::nd_range<2>{{n_tiles*block_rows, n_tiles*tile_size}, {block_rows, tile_size}},
        [=](sycl::nd_item<2> item) {
        const unsigned thX = item.get_local_id(1);
        const unsigned thY = item.get_local_id(0);
        unsigned column = item.get_group(1) * tile_size + thX;
        unsigned row    = item.get_group(0) * tile_size + thY;

	for (unsigned j = 0; j < tile_size; j += block_rows)
          if(row + j < m) tile[thY+j][thX] = in[(row+j)*lda + column];

        item.barrier(sycl::access::fence_space::local_space);

        column = item.get_group(0)*tile_size + thX;
        if(column<m)
        {
          row = item.get_group(1)*tile_size + thY;
          for (unsigned j = 0; j < tile_size; j += block_rows)
          if(row+j < n) out[(row + j)*ldb + column] = tile[thX][thY + j];
        }
    });
  });
}

int main(int argc, char** argv)
{
  sycl::queue m_queue;

  using T = double;
  constexpr size_t ALIGN = 64;
  int n = 31;
  int m = 33;
  T* A = sycl::aligned_alloc_device<T>(ALIGN, n * m, m_queue);
  T* B = sycl::aligned_alloc_device<T>(ALIGN, n * m, m_queue);

  std::vector<T> a(m*n);
  std::vector<T> b(m*n);

  std::iota(a.begin(), a.end(), 0);
  m_queue.memcpy(A, a.data(), a.size()*sizeof(T)).wait();

  transpose(m_queue, A, m, n, B, n, m).wait();
  m_queue.memcpy(b.data(), B, b.size()*sizeof(T)).wait();

  for(int i=0; i < m; ++i)
  {
    std::cout << i << " : ";
    for(int j = 0; j < n; ++j)
      std::cout << a[i*n + j] << " ";
    std::cout << std::endl;
  }

  for(int i=0; i < n; ++i)
  {
    std::cout << i << " : ";
    for(int j = 0; j < m; ++j)
      std::cout << b[i*m + j] << " ";
    std::cout << std::endl;
  }

  sycl::free(B, m_queue);
  sycl::free(A, m_queue);

}
