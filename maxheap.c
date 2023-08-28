#include <stdio.h>
#include <stdlib.h>
#include "maxheap.h"

/* initialize a heap */
void maxheap_init(maxheap_type* heap_ptr, int max_size) {
    heap_ptr->max_size = max_size;
    heap_ptr->array = (key_value_type*)calloc(max_size,sizeof(key_value_type));
    heap_ptr->size = 0;
}

/* free up a heap */
void maxheap_deinit(maxheap_type* heap_ptr) {
    heap_ptr->max_size = 0;
    free(heap_ptr->array);
    heap_ptr->array = 0;
    heap_ptr->size = 0;
}

/* insert a key/value pair into our heap */
void maxheap_insert(maxheap_type* heap_ptr, key_value_type new) {
    int done_heapify = 0;
    int current_index, parent_index;
    key_value_type temp;
    
    /* check to see if there is room for the new key/value pair */
    if (heap_ptr->size == heap_ptr->max_size) {
	printf("Error: heap already has max_size=%d key/value pairs!\n",
	       heap_ptr->max_size);
	exit (1); /* terminate program abnormally */
    }
    
    /* set the current index */
    current_index = heap_ptr->size; 
    
    /* insert at the end */  
    heap_ptr->array[current_index] = new; 
    heap_ptr->size += 1;
    
    /* if heap contains 1 node we are done since it is already a maxheap */  
    if (heap_ptr->size == 1) {
	done_heapify = 1;
    }
    
    /* climb the tree swapping as necessary to maintain maxheap property */
    while (done_heapify == 0) {
	parent_index = (current_index-1)/2;
	/* check to see if we need to swap value with parent */
	if (heap_ptr->array[parent_index].value < heap_ptr->array[current_index].value) {
	    temp = heap_ptr->array[current_index];
	    heap_ptr->array[current_index] = heap_ptr->array[parent_index];
	    heap_ptr->array[parent_index] = temp;
	} else {
	    /* done if parent value is greater than or equal to current */
	    done_heapify = 1;
	}
	/* climb the tree */
	current_index = parent_index;
	/* if we are at root we are done */
	if (current_index == 0) {
	    done_heapify = 1;
	}
    }
}

/* find the index of the largest child value */
/* we assume that there is at least one child */
int index_of_largest_child(maxheap_type* heap_ptr, int current_index) {
    int left_index, right_index;
    left_index = current_index*2+1;
    right_index = current_index*2+2;
    /* handle the single child case */
    if (right_index > heap_ptr->size-1) {
	return left_index;
    }
    /* handle the two child case */
    if (heap_ptr->array[right_index].value > heap_ptr->array[left_index].value) {
	return right_index;
    } else {
	return left_index;
    }
}

/* remove the largest value (root value) from the heap */
void maxheap_remove_root(maxheap_type* heap_ptr) {
    int done_heapify = 0;
    int current_index, child_index;
    key_value_type temp;
    
    /* nothing to do for an empty heap */
    if (heap_ptr->size == 0) {
	return;
    }
    
    /* handle the case where the heap has one element */
    if (heap_ptr->size == 1) {
	heap_ptr->size = 0;
	return;
    }
    
    /* move the last element to the root and decrease size by 1 */
    heap_ptr->array[0] = heap_ptr->array[heap_ptr->size-1];
    heap_ptr->size -= 1;
    
    /* start at the root */
    current_index = 0;
    
    /* if heap contains 1 node we are done since it is already a maxheap */
    if (heap_ptr->size == 1) {
	return;
    }
    
    /* descend the tree swapping as necessary to maintain maxheap prop. */
    while (done_heapify == 0) {
	child_index = index_of_largest_child(heap_ptr,current_index);
	if (heap_ptr->array[current_index].value < heap_ptr->array[child_index].value) {
	    temp = heap_ptr->array[current_index];
	    heap_ptr->array[current_index] = heap_ptr->array[child_index];
	    heap_ptr->array[child_index] = temp;
	} else {
	    /* done if largest child value is less than or equal to ours */
	    done_heapify = 1;
	}
	/* descend the tree */
	current_index = child_index;
	/* if there are no children, we are done */
	if (current_index*2+1 > heap_ptr->size-1) {
	    done_heapify = 1;
	}
    }
}
