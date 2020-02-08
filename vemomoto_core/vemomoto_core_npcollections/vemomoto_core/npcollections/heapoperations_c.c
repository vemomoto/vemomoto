/*
 * _chi2.c
 *
 *  Created on: 26.11.2016
 *      Author: Samuel
 */

#include <stdio.h>

/* Available functions */
static void heapifyUp(double *heap, long *keys, long *positions, long start);
static void heapifyDown(double *heap, long *keys, long *positions, long start,
		long size);

void heapifyUp(double *heap, long *keys, long *positions, long start) {
	long vertex, parent, key;
	double priority;
	vertex = start;
	key = keys[vertex];
	priority = heap[vertex];

	/* debug
	printf("Start: %i \n", vertex);
	printf("Key: %i \n", key);
	printf("Priority: %f \n", priority);
	*/

	while (vertex) {
		parent = (vertex-1) / 2;

		/* debug
		printf("B_");
		printf("Parent:%i ", parent);
		printf("ParentKey:%i ", keys[parent]);
		printf("ParentPrior:%f\n", heap[parent]);
		 */

		/* if heap property is not satisfied */
		if (priority < heap[parent]) {
			/* swap vertex and parent */

			/* debug
			printf("C >> ");
			printf("%i ", parent);
			printf("%i ", keys[parent]);
			printf("%i ", vertex);
			printf("%f ", heap[parent]);
			printf("%f ", priority);
			printf("swap\n");
			 */

			positions[keys[parent]] = vertex;
			heap[vertex] = heap[parent];
			keys[vertex] = keys[parent];
			vertex = parent;
		} else {
			break;
		}
	}

	if (start != vertex) {
		/* debug
		printf("E  ");
		printf("%i ", key);
		printf("%i \n", vertex);
		printf("%i \n", positions[key]);
		*/

		positions[key] = vertex;
		heap[vertex] = priority;
		keys[vertex] = key;
	}

}

void heapifyDown(double *heap, long *keys, long *positions,
		long start, long size) {
	long vertex, child, key;
	double priority;

	vertex = start;
	key = keys[vertex];
	priority = heap[vertex];
	while (1) {
		child = 2 * vertex + 1;
		if (child >= size) {
			break;
		}

		if (child + 1 < size && heap[child + 1] < heap[child]) {
			child++;
		}

		/* if heap property is not satisfied */
		if (priority > heap[child]) {
			/* swap vertex and parent */

			/* debug
			printf("%f  ", heap[child]);
			printf("%f  ", priority);
			printf("swap\n");
			 */

			positions[keys[child]] = vertex; /*positions[keys[vertex]];*/
			heap[vertex] = heap[child];
			keys[vertex] = keys[child];
			vertex = child;
		} else {
			break;
		}
	}

	if (vertex != start) {
		positions[key] = vertex;
		heap[vertex] = priority;
		keys[vertex] = key;
	}

}
