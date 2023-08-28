#include <stdio.h>
#include <stdlib.h> 
#include <mpi.h>
#include "matrix.h"
#include "maxheap.h"

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

/* classify a test image given the k nearest neighbors */
/* predict the class using the "majority rule" */
/* if there is a tie reduce k by 1 and repeat until a single class has a majority */
/* note that the tie breaking process is guaranteed to terminate when k=1 */
int classify (matrix_type* train_labels_ptr, int num_classes, int k, int* knearest) {
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

    int rank, size;

    MPI_Init(&argc, &argv);

    /* start the timer */
    double start, end;
    start = MPI_Wtime();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* get max_k and num_to_test from the command line */
    if (argc != 3) {
	printf ("Command usage : %s %s %s\n",argv[0],"max_k","num_to_test");
	return 1;
    }
    int max_k = atoi(argv[1]);
    int num_to_test = atoi(argv[2]);
    if (num_to_test > 10000) num_to_test = 10000;

    /* the MNIST dataset has 10 class labels */
    int num_classes = 10;

    /* read in the mnist training set of 1000000 images and labels */
    int num_train = 1000000;
    matrix_type train_images, train_labels;
    matrix_init (&train_images,num_train,784);
    matrix_read_bin(&train_images,"mnist1m-images-idx3-ubyte",16);
    matrix_init (&train_labels,num_train,1);
    matrix_read_bin(&train_labels,"mnist1m-labels-idx1-ubyte",8);
    
    /* read in the mnist test set of 10000 images */
    int num_test = 10000;
    matrix_type test_images, test_labels;
    matrix_init (&test_images,num_test,784);
    matrix_read_bin(&test_images,"t10k-images-idx3-ubyte",16);
    matrix_init (&test_labels,num_test,1);
    matrix_read_bin(&test_labels,"t10k-labels-idx1-ubyte",8);

    /* keep track of the number of images misclassified for each k */
    int num_missed[max_k];
    for (int k=1;k<=max_k;k++) {
	num_missed[k-1] = 0;
    }

    /* for each test image: */
    /*  find the max_k training images nearest the given test image */
    /*  For each value of k from 1 to max_k: */
    /*   classify the test image and keep track of the number missed */
    int i,k;
    matrix_row_type test_image;
    int knearest[max_k];
    int predicted_label;
    for (i = 0;i<num_to_test;i++) {
	matrix_get_row(&test_images,&test_image,i);
	find_knearest (&train_images,&test_image,max_k,knearest);
	for (k=1;k<=max_k;k++) {
	    predicted_label = classify (&train_labels,num_classes,k,knearest);
	    if (predicted_label != test_labels.data_ptr[i]) {
		num_missed[k-1] += 1;
	    }
	}
    }

    /* print the results */
    int min_num_missed = num_to_test+1;
    int best_k = 0;
    for (int k=1;k<=max_k;k++) {
	if (num_missed[k-1] < min_num_missed) {
	    min_num_missed = num_missed[k-1];
	    best_k = k;
	}
    }
    float accuracy = (float)(num_to_test-min_num_missed)/num_to_test;
    printf ("best_k = %d, num_missed = %d, accuracy = %g\n",
	    best_k, min_num_missed, accuracy);
	
    /* free up the training and test data sets */
    matrix_deinit(&train_images);
    matrix_deinit(&test_images);
    matrix_deinit(&train_labels);
    matrix_deinit(&test_labels);

    /* stop timer and print wall time used */
    end = MPI_Wtime();
    if (rank==0) {
	printf ("wall time used = %g sec\n",end-start);
    }
    
    MPI_Finalize();

    return 0;
}

