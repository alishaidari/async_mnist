/* maxheap.h */

/* make sure this header file has not been included already */
#ifndef __MAXHEAP
#define __MAXHEAP 1

/* key/value pair */
typedef struct key_value_s {
    int key;
    int value;
} key_value_type;

/* The maxheap structure */
typedef struct maxheap_s {
    int max_size;
    key_value_type* array;
    int size;
} maxheap_type;

/* initialize a heap */
void maxheap_init(maxheap_type* heap_ptr, int max_size);

/* free up a heap */
void maxheap_deinit(maxheap_type* heap_ptr);

/* insert a key/value pair into our heap */
void maxheap_insert(maxheap_type* heap_ptr, key_value_type new);

/* remove the largest value (root value) from the heap */
void maxheap_remove_root(maxheap_type* heap_ptr);

#endif
