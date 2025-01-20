//@author RhinoCoder
//Term Project
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <papi.h>
#include <limits.h>

void GenerateElevenBandedCsrLocal(long long rank, long long N,long long offset, long long localN,long long **rowPtr, long long **colInd,double **V)
{
    long long haloStart = (offset - 5 < 0) ? 0 : offset - 5;
    long long haloEnd = ((offset + localN - 1) + 5 >= N)
                             ? (N - 1)
                             : (offset + localN - 1 + 5);

    if (haloStart < 0 || haloEnd >= N)
    {
        fprintf(stderr, "[Rank %lld] Halo region out of bounds: [%lld, %lld]\n",rank, haloStart, haloEnd);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    *rowPtr = (long long *)malloc((localN + 1) * sizeof(long long));
    if (!(*rowPtr))
    {
        fprintf(stderr, "[Rank %lld] Error allocate rowPtr.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    long long *nnzPerRow = (long long *)calloc(localN, sizeof(long long));
    if (!nnzPerRow)
    {
        fprintf(stderr, "[Rank %lld] Error allocate nnzPerRow.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    long long i_local;
    for (i_local = 0; i_local < localN; i_local++)
    {
        long long i_global = offset + i_local;

        long long start_col = (i_global - 5 < 0) ? 0 : (i_global - 5);
        long long end_col = (i_global + 5 >= N) ? (N - 1) : (i_global + 5);
        long long clipped_start = (start_col < haloStart) ? haloStart : start_col;
        long long clipped_end = (end_col > haloEnd) ? haloEnd : end_col;

        if (clipped_end >= clipped_start)
        {
            nnzPerRow[i_local] = clipped_end - clipped_start + 1;
        }
        else
        {
            nnzPerRow[i_local] = 0;
        }
    }

    (*rowPtr)[0] = 0;
    long long i;
    for (i = 0; i < localN; i++)
    {
        (*rowPtr)[i + 1] = (*rowPtr)[i] + nnzPerRow[i];
    }

    long long localNNZ = (*rowPtr)[localN];

    *colInd = (long long *)malloc(localNNZ * sizeof(long long));
    *V = (double *)malloc(localNNZ * sizeof(double));
    if (!(*colInd) || !(*V))
    {
        fprintf(stderr, "@[Rank %lld] Error allocate colInd/val.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    long long i_local2;
    for (i_local2 = 0; i_local2 < localN; i_local2++)
    {
        long long i_global = offset + i_local2;
        long long start_col = (i_global - 5 < 0) ? 0 : (i_global - 5);
        long long end_col = (i_global + 5 >= N) ? (N - 1) : (i_global + 5);
        long long clipped_start = (start_col < haloStart) ? haloStart : start_col;
        long long clipped_end = (end_col > haloEnd) ? haloEnd : end_col;
        long long row_start = (*rowPtr)[i_local2];
        long long row_len = nnzPerRow[i_local2];

        long long k;
        for (k = 0; k < row_len; k++)
        {
            long long j_global = clipped_start + k;
            (*colInd)[row_start + k] = j_global;
            (*V)[row_start + k] = (double)(rand() % 1000);
        }
    }

    free(nnzPerRow);
}

void SpmvMCsrLocally(long long localN, long long offset, long long haloStart, long long localXSize, const long long *rowPtr, const long long *colInd, const double *V, const double *subX, double *yLocal)
{
    long long i_local;
    for (i_local = 0; i_local < localN; i_local++)
    {
        double sum = 0.0;
        long long start = rowPtr[i_local];
        long long end = rowPtr[i_local + 1];

        long long k;
        for (k = start; k < end; k++)
        {
            long long j_global = colInd[k];
            long long jExtended = j_global - haloStart + 5;
            if (jExtended >= 0 && jExtended < localXSize)
            {
                sum += V[k] * subX[jExtended];
            }
        }
        yLocal[i_local] = sum;
    }
}

int main(int argc, char *argv[])
{

    int rankInt, sizeInt;

    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankInt);
    MPI_Comm_size(MPI_COMM_WORLD, &sizeInt);

    long long rank = (long long)rankInt;
    long long size = (long long)sizeInt;

    if (argc < 2)
    {
        if (rank == 0)
        {
            printf("Usage: mpirun -n <procs> %s <N>\n", argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    long long N = atoll(argv[1]);
    if (N <= 0)
    {
        if (rank == 0)
        {
            fprintf(stderr, "Error: N must be > 0.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    FILE *fp = NULL;
    if (rank == 0)
    {
        fp = fopen("TermProjectResults.txt", "w");
        if (!fp)
        {
            fprintf(stderr, "Could not open output .txt file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fprintf(fp, "_____________________\n");
        fprintf(fp, "Term Project:\n");
        fprintf(fp, "_____________________\n\n");
        fprintf(fp, "> Eleven Banded Non Symmetric Sparse Matrix - Dense Vector Multiplication \n\n");
        fprintf(fp, "N = %lld, size = %lld\n", N, size);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    long long base = N / size;
    long long rem = N % size;
    long long localN = base + ((rank < rem) ? 1 : 0);

    if (localN == 0)
    {
        //If no process is assigned, terminate normally without cause any error.
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        return 0;
    }

    long long offset = 0;
    long long r;
    for (r = 0; r < rank; r++)
    {
        offset += base + ((r < rem) ? 1 : 0);
    }

    srand((unsigned)time(NULL) + (unsigned)rank * 5843);

    long long *row_ptr_local = NULL;
    long long *col_ind_local = NULL;
    double *val_local = NULL;
    
    GenerateElevenBandedCsrLocal(rank, N, offset, localN, &row_ptr_local, &col_ind_local, &val_local);
    MPI_Barrier(MPI_COMM_WORLD);
    
    long long haloStart = (offset - 5 < 0) ? 0 : (offset - 5);
    long long haloEnd = ((offset + localN - 1) + 5 >= N) ? (N - 1) : (offset + localN - 1 + 5);
    long long localXSize = haloEnd - haloStart + 1;
    
    if (localXSize <= 0)
    {
        localXSize = 0;
    }

    double *subX = (double *)malloc(localXSize * sizeof(double));
    if (!subX)
    {
        fprintf(stderr, "Rank:%lld Cannot allocate x_sub of size %lld\n", rank, localXSize);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    long long i;
    for (i = 0; i < localXSize; i++)
    {
        subX[i] = (double)(rand() % 1000);
    }

    double *yLocal = (double *)calloc(localN, sizeof(double));
    if (!yLocal)
    {
        fprintf(stderr, " Failed to allocate yLocal @ Rank: %lld\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
 

    long long left_neighbor = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    long long right_neighbor = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    double leftHalo[5] = {0};
    double rightHalo[5] = {0};
    double leftSend[5] = {0};
    double rightSend[5] = {0};

    if (localXSize >= 5)
    {
        int haloSend;
        for (haloSend = 0; haloSend < 5; haloSend++)
        {
            if (offset > 0)
            {
                leftSend[haloSend] = subX[haloSend];
            }
            else
            {
                leftSend[haloSend] = 0.0;
            }

            if (offset + localN < N)
            {
                rightSend[haloSend] = subX[localXSize - 5 + haloSend];
            }
            else
            {
                rightSend[haloSend] = 0.0;
            }
        }
    }
 

    int rightCode = MPI_Sendrecv(rightSend, 5, MPI_DOUBLE, right_neighbor, 0, leftHalo, 5, MPI_DOUBLE, left_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (rightCode != MPI_SUCCESS)
    {
        char errMsg[MPI_MAX_ERROR_STRING];
        int msgLen;
        MPI_Error_string(rightCode, errMsg, &msgLen);
        fprintf(stderr, "error in MPI_Sendrcv for right send at rank %lld: %s\n", rank, errMsg);
        MPI_Abort(MPI_COMM_WORLD, rightCode);
    }

    int leftCode = MPI_Sendrecv(leftSend, 5, MPI_DOUBLE, left_neighbor, 1, rightHalo, 5, MPI_DOUBLE, right_neighbor, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (leftCode != MPI_SUCCESS)
    {
        char errMsg[MPI_MAX_ERROR_STRING];
        int msgLen;
        MPI_Error_string(leftCode, errMsg, &msgLen);
        fprintf(stderr, "error in MPI_Sendrcv for left send at rank %lld: %s\n", rank, errMsg);
        MPI_Abort(MPI_COMM_WORLD, leftCode);
    }
 
    

    MPI_Barrier(MPI_COMM_WORLD);
    double *x_extended = (double *)malloc((localXSize + 10) * sizeof(double));
    if (!x_extended)
    {
        fprintf(stderr, "error x_extended at @Rank %lld\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    memcpy(x_extended, leftHalo, 5 * sizeof(double));
    memcpy(x_extended + 5, subX, localXSize * sizeof(double));
    memcpy(x_extended + 5 + localXSize, rightHalo, 5 * sizeof(double));

    SpmvMCsrLocally(localN, offset, haloStart, localXSize + 10, row_ptr_local, col_ind_local, val_local, x_extended, yLocal);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double localPassedTime = (t1 - t0) * 1000.0;

    double *yGlobal = NULL;
    int *recvCounts = NULL;
    int *displs = NULL;

    if (rank == 0)
    {
        yGlobal = (double *)malloc(N * sizeof(double));
        recvCounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));

        if (!yGlobal)
        {
            fprintf(stderr, "Memory allocation failed on rank 0 for yGlobal (size=%lld)\n", N);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (!recvCounts)
        {
            fprintf(stderr, "Memory allocation failed on rank 0 for recvcounts (size=%lld)\n", N);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (!displs)
        {
            fprintf(stderr, "Memory allocation failed on rank 0 for displs (size=%lld)\n", N);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        long long offsetR = 0;
        long long rr;
        for (rr = 0; rr < size; rr++)
        {
            long long localNrr = base + ((rr < rem) ? 1 : 0);
            recvCounts[rr] = (int)localNrr;
            displs[rr] = (int)offsetR;
            offsetR += localNrr;
        }
    }

    if (rank == 0)
    {
        // Debugging session of the code.
        // If you awant to avoid overheads that coming from printfs, delete them. I kept
        // them during projcet because wanted to be sure about the computation.
        int rr;
        for (rr = 0; rr < size; rr++)
        {
            printf("@[Rank %lld] recvCounts[%d]: %d, displs[%d]: %d\n", rank, rr, recvCounts[rr], rr, displs[rr]);
            fprintf(fp, "@[Rank %lld] recvCounts[%d]: %d, displs[%d]: %d\n", rank, rr, recvCounts[rr], rr, displs[rr]);
        }

        long long totalRows = 0;
        int ii;
        for (ii = 0; ii < size; ii++)
        {
            totalRows += recvCounts[ii];
        }
        if (totalRows != N)
        {
            fprintf(stderr, "@[Rank: %lld] Total recvcounts (%lld) != N (%lld)\n", rank, totalRows, N);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int gCode = MPI_Gatherv(yLocal, localN, MPI_DOUBLE, yGlobal, recvCounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (gCode != MPI_SUCCESS)
    {
        char errMsg[MPI_MAX_ERROR_STRING];
        int msgLen;
        MPI_Error_string(gCode, errMsg, &msgLen);
        fprintf(stderr, "Error in MPI_Gatherv @ rank %lld: %s\n", rank, errMsg);
        MPI_Abort(MPI_COMM_WORLD, gCode);
    }

    if (rank == 0)
    {
        fprintf(fp, "> Showing results.....\n\n");
        fprintf(fp, "yGlobal (first 5 entries): ");

        long long i;
        for (i = 0; i < 5 && i < N; i++)
        {
            fprintf(fp, "%.2f ", yGlobal[i]);
        }
        fprintf(fp, "\n");
        fprintf(fp, "...............................................\n");
        fprintf(fp, ".....SpMV (Row-block) with N=%lld, Proc Count=%lld .....\n", N, size);
        fprintf(fp, "...............................................\n\n");
        fprintf(fp, "Elapsed time @ [rank 0]: %.3f ms\n\n\n", localPassedTime);
        fprintf(fp, "...............................................\n");
        fprintf(fp, "PAPI Profiling Results\n");
        fprintf(fp, "...............................................\n\n");
        //Insert papi results here, for base code it is not included in here,
        //A papi inserted version of the code is provided, please check it out.
        free(yGlobal);
        free(recvCounts);
        free(displs);
        fclose(fp);
    }

    free(yLocal);
    free(subX);
    free(row_ptr_local);
    free(col_ind_local);
    free(val_local);

    MPI_Finalize();
    return 0;
}
