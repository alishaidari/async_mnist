#include <stdio.h>
#include <stdlib.h> 
#include <omp.h>
#include "matrix.h"
#include "maxheap.h"

/* find the k nearest neighbors sorted from closest to farthest */
void find_knearest (matrix_type* train_images_ptr, matrix_row_type* test_image_ptr,
		    int k, int* knearest) {

    matrix_row_type train_image;
    key_value_type pair;
    maxheap_type heap;
    maxheap_init (&heap,k);
    int i;
    for (i = 0;i<train_images_ptr->num_rows;i++) {
	matrix_get_row(train_images_ptr,&train_image,i);
	pair.key = i;
	pair.value = matrix_row_dist_sq(test_image_ptr,&train_image);
	if (heap.size < k) {
	    maxheap_insert(&heap,pair);      
	} else if (pair.value < heap.array[0].value) {
	    maxheap_remove_root(&heap);
	    maxheap_insert(&heap,pair);
	}
    }
    
    /* store the k nearest neighbors from closest to farthest */
    for (i = k-1; i>=0; i--) {
	    pair = heap.array[0];
	    knearest[i] = pair.key;
	    maxheap_remove_root(&heap);
    } 
    /* free up the heap */
    maxheap_deinit(&heap);
}

/* classify a test image given the k nearest neighbor indices */
/* predict the class of a test image using the "majority rule" */
/* if there is a tie reduce k by 1 and repeat until a single class has a majority */
/* note that the tie breaking process is guaranteed to terminate when k=1 */
int classify (matrix_type* train_labels_ptr, int num_classes, int k, int* knearest){
     
    //dynamically allocate knearest label buffer
    int* knearest_label = calloc(k, sizeof(int)); 
    int class = -100;
    //fill in buffer for knearest label arr
    for (int i=0;i<k;i++){
        knearest_label[i] = train_labels_ptr->data_ptr[knearest[i]];
        //printf("%d ", knearest_label[i]);
    }
    //printf("\n");       
    int count = 0; 
    int majority_idx = -1;
    int majority_count = 0;  
    int majority_candidate = 0;
    int num_of_majorities = 0;
    int majority_flag = 1;
    while (majority_flag == 1 && k != 1){
       for (int i=0; i<k; i++){ 
            count = 0;
            for (int j=0;j<k;j++){
                if (knearest_label[i] == knearest_label[j]){
                    count++;
                }
            }
            if (count > 1){
                num_of_majorities++;
            }
            if(count > majority_count){
                majority_count = count;
                majority_idx = i;
            }
            //printf("index: %d , freq: %d\n", i, count);
        }
        num_of_majorities = num_of_majorities/2;
        /*
        printf("majority_idx: %d\n", majority_idx);
        printf("majority_candidate: %d\n", knearest_label[majority_idx]);
        printf("majority_freq: %d\n", majority_count);
        printf("current_k: %d\n", k);
        printf("num_of_majorities: %d\n", num_of_majorities);
        printf("\n");
        */
        //check if we have to reduce k or not
        if (num_of_majorities == 1 || majority_count > (k/2)){
            class = knearest_label[majority_idx];
            majority_flag = -1;
        }
        else {
            k--;
            num_of_majorities = 0;
            majority_count = 0;
        }
    }

    return class;

    //free dynamically allocated buffer    
    free(knearest_label); 
}

int main (int argc, char** argv) {

    /* get k, start test image, and num_to_test from the command line */
    if (argc != 4) {
	printf ("Command usage : %s %s %s %s\n",argv[0],"k","start_test","num_to_test");
	return 1;
    }
    int k = atoi(argv[1]);
    int start_test = atoi(argv[2]);
    int num_to_test = atoi(argv[3]);
    if (num_to_test + start_test > 10000) num_to_test = 10000-start_test;

    /* the MNIST dataset has 10 class labels */
    int num_classes = 10;
    
    /* read in the mnist training set of 60000 images and labels */
    int num_train = 60000;
    matrix_type train_images, train_labels;
    matrix_init (&train_images,num_train,784);
    matrix_read_bin(&train_images,"train-images-idx3-ubyte",16);
    matrix_init (&train_labels,num_train,1);
    matrix_read_bin(&train_labels,"train-labels-idx1-ubyte",8);
    
    /* read in the mnist test set of 10000 images */
    int num_test = 10000;
    matrix_type test_images, test_labels;
    matrix_init (&test_images,num_test,784);
    matrix_read_bin(&test_images,"t10k-images-idx3-ubyte",16);
    matrix_init (&test_labels,num_test,1);
    matrix_read_bin(&test_labels,"t10k-labels-idx1-ubyte",8);

    /* find the k training images nearest the given test image */
    int i,j;
    matrix_row_type test_image;
    int knearest[k];
    int predicted_label;
    for (i = start_test;i<start_test+num_to_test;i++) {
	matrix_get_row(&test_images,&test_image,i);
	find_knearest (&train_images,&test_image,k,knearest);
	predicted_label = classify (&train_labels,num_classes,k,knearest);
	printf ("test index : %d, test label : %d, ",
		i,test_labels.data_ptr[i]);
	printf ("training labels : ");
	for (j = 0;j<k;j++) {
	    printf ("%d ",train_labels.data_ptr[knearest[j]]);
	}
	printf (", predicted label : %d",predicted_label);
	printf ("\n");
    }

    /* free up the training and test data sets */
    matrix_deinit(&train_images);
    matrix_deinit(&test_images);
    matrix_deinit(&train_labels);
    matrix_deinit(&test_labels);

    return 0;
}
