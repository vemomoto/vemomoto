/*
 *
 *
 *  Created on: 26.11.2016
 *      Author: Samuel
 */

#include <stdio.h>

/* Available functions */
static void sectionsum(double *arr, long long *indptr, long long arrzise, long long ptrsize, double *out);
static void sectionsum_chosen(double *arr, long long *arrindptr, long long *consd, long long *consdindptr, long long rownumber, double *out);
static void sectionsum_chosen_rows(double *arr, long long *arrindptr, long long *consd,
		long long *consdindptr, long long *rows, long long rownumber, double *out);
static void sectionsum_chosen_rows_fact(double *arr, long long *arrindptr, long long *consd,
		long long *consdindptr, long long *rows, double *factor, long long rownumber, double *out);
static void sectionprod(double *arr, long long *indptr, long long ptrsize, double *out);
static void sectionsum_rowprod(double *arr1, long long *arr1indptr, long long *columns1,
		long long *columns1indptr, long long *columns1rows, long long *rows1,
		double *arr2, long long *arr2indptr, long long *rows2,
		long long outsize, double *out);

void sectionsum(double *arr, long long *indptr, long long arrzise, long long ptrsize, double *out) {
	int rowIndex;
	int ptr;
	int i;
	double tmp;

	i=0;
	for (rowIndex=1; rowIndex<ptrsize; rowIndex++) {
		tmp = 0;
		ptr = indptr[rowIndex];
		/*printf("ptr: %i \n", ptr);
		printf("i: %i \n", i);
			printf("arr[i]: %f \n", arr[i]);
		printf("tmp: %f \n", tmp);
		*/
		for (i=i; i<ptr; i++) {
			tmp = tmp + arr[i];
		}
		out[rowIndex-1] = tmp;
	}

}

void sectionsum_chosen(double *arr, long long *arrindptr, long long *consd, long long *consdindptr, long long rownumber, double *out) {
	int rowIndex;
	int consptr;
	int addindex;
	int i;
	double tmp;

	i=0;
	for (rowIndex=0; rowIndex<rownumber-1; rowIndex++) {
		tmp = 0;
		consptr = consdindptr[rowIndex+1];
		addindex = arrindptr[rowIndex];
		/*printf("ptr: %i \n", ptr);
		printf("i: %i \n", i);
			printf("arr[i]: %f \n", arr[i]);
		printf("tmp: %f \n", tmp);
		*/
		for (i=i; i<consptr; i++) {
			tmp = tmp + arr[addindex+consd[i]];
		}
		out[rowIndex] = tmp;
	}

}

void sectionsum_chosen_rows(double *arr, long long *arrindptr, long long *consd,
		long long *consdindptr, long long *rows, long long rownumber, double *out) {
	int rowIndex;
	int consptr;
	int addindex;
	int i;
	double tmp;

	i=0;
	for (rowIndex=0; rowIndex<rownumber; rowIndex++) {
		tmp = 0;
		consptr = consdindptr[rowIndex+1];
		addindex = arrindptr[rows[rowIndex]];
		/*printf("ptr: %i \n", ptr);
		printf("rowIndex: %i \n", rowIndex);
		printf("rows[rowIndex]: %i \n", rows[rowIndex]);
		printf("arrindptr[rows[rowIndex]]: %i \n", arrindptr[rows[rowIndex]]);
		printf("i: %i \n", i);
			printf("arr[i]: %f \n", arr[i]);
		printf("tmp: %f \n", tmp);
		*/
		for (i=i; i<consptr; i++) {
			tmp = tmp + arr[addindex+consd[i]];
		}
		out[rowIndex] = tmp;
	}

}


void sectionsum_chosen_rows_fact(double *arr, long long *arrindptr, long long *consd,
		long long *consdindptr, long long *rows,
		double *factor, long long rownumber, double *out) {
	int rowIndex;
	int consptr;
	int addindex;
	int i;
	double tmp;

	i=0;
	for (rowIndex=0; rowIndex<rownumber; rowIndex++) {
		tmp = 0;
		consptr = consdindptr[rowIndex+1];
		addindex = arrindptr[rows[rowIndex]];
		for (i=i; i<consptr; i++) {
			tmp = tmp + arr[addindex+consd[i]] * factor[i];
		}
		out[rowIndex] = tmp;
	}

}

void sectionsum_rowprod(double *arr1, long long *arr1indptr, long long *columns1,
		long long *columns1indptr, long long *columns1rows, long long *rows1,
		double *arr2, long long *arr2indptr, long long *rows2,
		long long outsize, double *out) {
	int rowIndex;
	int columns1indptrRowIndex;
	int columns1RowStart;
	int columns1RowEnd;
	int arr1RowStart;
	int arr2RowStart;
	int i;
	double tmp;

	for (rowIndex=0; rowIndex<outsize; rowIndex++) {
		tmp = 0;
		columns1indptrRowIndex = columns1rows[rowIndex];
		columns1RowStart = columns1indptr[columns1indptrRowIndex];
		columns1RowEnd = columns1indptr[columns1indptrRowIndex+1];
		arr1RowStart = arr1indptr[rows1[rowIndex]];
		arr2RowStart = arr2indptr[rows2[rowIndex]];
		/*printf("rowIndex: %i \n", rowIndex);
		printf("arr1RowStart: %i \n", arr1RowStart);
		printf("arr2RowStart: %i \n", arr2RowStart);
		printf("rows2[rowIndex]: %i \n", rows2[rowIndex])*/;

		for (i=0; i<columns1RowEnd-columns1RowStart; i++) {
			/*printf("..i: %i \n", i);
			printf("..columns1[columns1RowStart+i]: %i \n", columns1[columns1RowStart+i]);
			printf("..arr1[arr1RowStart+columns1[columns1RowStart+i]]: %f \n", arr1[arr1RowStart+columns1[columns1RowStart+i]]);
			printf("..arr2[arr2RowStart+columns1[columns1RowStart+i]]: %f \n", arr2[arr2RowStart+columns1[columns1RowStart+i]]);*/
			tmp = tmp + arr1[arr1RowStart+columns1[columns1RowStart+i]]
							 * arr2[arr2RowStart+columns1[columns1RowStart+i]];
		}
		out[rowIndex] = tmp;
	}

}


void sectionprod(double *arr, long long *indptr, long long ptrsize, double *out) {
	int rowIndex;
	int ptr;
	int i;
	double tmp;

	i=0;
	for (rowIndex=1; rowIndex<ptrsize; rowIndex++) {
		tmp = 1;
		ptr = indptr[rowIndex];
		/*printf("ptr: %i \n", ptr);
		printf("i: %i \n", i);
			printf("arr[i]: %f \n", arr[i]);
		printf("tmp: %f \n", tmp);
		*/
		for (i=i; i<ptr; i++) {
			tmp = tmp * arr[i];
		}
		out[rowIndex-1] = tmp;
	}

}
