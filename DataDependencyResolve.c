#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <papi.h>
#include <limits.h>

void GenerateElevenBandedCsrLocal(long long rank, long long N,
                                  long long offset, long long local_n,
                                  long long **row_ptr, long long **col_ind,
                                  double **val)
{
    // ----------------------------------------------------------------------
    // 1) Compute halo_start, halo_end right away so we can do proper clipping.
    // ----------------------------------------------------------------------
    long long halo_start = (offset - 5 < 0) ? 0 : offset - 5;
    long long halo_end = ((offset + local_n - 1) + 5 >= N)
                             ? (N - 1)
                             : (offset + local_n - 1 + 5);

    // Optional safety check
    if (halo_start < 0 || halo_end >= N)
    {
        fprintf(stderr, "[Rank %lld] Halo region out of bounds: [%lld, %lld]\n",
                rank, halo_start, halo_end);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // ----------------------------------------------------------------------
    // 2) Allocate row_ptr and a temp array nnz_per_row (size local_n).
    // ----------------------------------------------------------------------
    *row_ptr = (long long *)malloc((local_n + 1) * sizeof(long long));
    if (!(*row_ptr))
    {
        fprintf(stderr, "[Rank %lld] Error: failed to allocate row_ptr.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    long long *nnz_per_row = (long long *)calloc(local_n, sizeof(long long));
    if (!nnz_per_row)
    {
        fprintf(stderr, "[Rank %lld] Error: failed to allocate nnz_per_row.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // ----------------------------------------------------------------------
    // 3) First pass: For each local row, compute how many columns
    //    actually lie in the clipped band [halo_start, halo_end].
    // ----------------------------------------------------------------------
    long long i_local;
    for (i_local = 0; i_local < local_n; i_local++)
    {
        long long i_global = offset + i_local;

        // Compute full 11-band range ...
        long long start_col = (i_global - 5 < 0) ? 0 : (i_global - 5);
        long long end_col = (i_global + 5 >= N) ? (N - 1) : (i_global + 5);

        // ... then clip to [halo_start, halo_end].
        long long clipped_start = (start_col < halo_start) ? halo_start : start_col;
        long long clipped_end = (end_col > halo_end) ? halo_end : end_col;

        // If there's no overlap, row has 0 columns
        if (clipped_end >= clipped_start)
        {
            nnz_per_row[i_local] = clipped_end - clipped_start + 1;
        }
        else
        {
            nnz_per_row[i_local] = 0;
        }
    }

    // ----------------------------------------------------------------------
    // 4) Build row_ptr[] as the prefix sum of nnz_per_row[].
    // ----------------------------------------------------------------------
    (*row_ptr)[0] = 0;
    long long i;
    for (i = 0; i < local_n; i++)
    {
        (*row_ptr)[i + 1] = (*row_ptr)[i] + nnz_per_row[i];
    }

    // local_nnz is row_ptr[local_n]
    long long local_nnz = (*row_ptr)[local_n];

    // ----------------------------------------------------------------------
    // 5) Allocate col_ind[] and val[] to match local_nnz
    // ----------------------------------------------------------------------
    *col_ind = (long long *)malloc(local_nnz * sizeof(long long));
    *val = (double *)malloc(local_nnz * sizeof(double));
    if (!(*col_ind) || !(*val))
    {
        fprintf(stderr, "[Rank %lld] Error: failed to allocate col_ind/val.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // ----------------------------------------------------------------------
    // 6) Second pass: Fill col_ind[] (and val[]) with the clipped columns.
    // ----------------------------------------------------------------------
    long long i_local2;
    for (i_local2 = 0; i_local2 < local_n; i_local2++)
    {
        long long i_global = offset + i_local2;
    
        long long start_col = (i_global - 5 < 0) ? 0 : (i_global - 5);
        long long end_col = (i_global + 5 >= N) ? (N - 1) : (i_global + 5);

        // Clip again
        long long clipped_start = (start_col < halo_start) ? halo_start : start_col;
        long long clipped_end = (end_col > halo_end) ? halo_end : end_col;

        // Where does this row begin in col_ind[]?
        long long row_start = (*row_ptr)[i_local2];
        // How many columns in this row?
        long long row_len = nnz_per_row[i_local2];

        // Fill them in
        long long k;
        for (k = 0; k < row_len; k++)
        {
            long long j_global = clipped_start + k;
            (*col_ind)[row_start + k] = j_global;
            (*val)[row_start + k] = (double)(rand() % 1000);
        }
    }

    free(nnz_per_row);
}

void SpmvMCsrLocally(long long local_n, long long offset, long long halo_start, long long local_x_size, const long long *row_ptr, const long long *col_ind, const double *val, const double *x_sub, double *yLocal)
{
    long long i_local;
    for (i_local = 0; i_local < local_n; i_local++)
    {
        double sum = 0.0;
        long long start = row_ptr[i_local];
        long long end = row_ptr[i_local + 1];

        long long k;
        for (k = start; k < end; k++)
        {
            long long j_global = col_ind[k];
            long long j_sub = j_global - halo_start;
            if (j_sub >= 0 && j_sub < local_x_size)
            {
                sum += val[k] * x_sub[j_sub];
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
    long long local_n = base + ((rank < rem) ? 1 : 0);
    
    if (local_n == 0){
    	// This rank has no rows. Skip everything safely:
    	// Possibly do an MPI_Barrier, a gather of zero elements, etc.
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
    GenerateElevenBandedCsrLocal(rank, N, offset, local_n, &row_ptr_local, &col_ind_local, &val_local);
    MPI_Barrier(MPI_COMM_WORLD);
    long long halo_start = (offset - 5 < 0) ? 0 : (offset - 5);
    long long halo_end = ((offset + local_n - 1) + 5 >= N) ? (N - 1) : (offset + local_n - 1 + 5);
    long long local_x_size = halo_end - halo_start + 1;
    if(local_x_size <=0){
     local_x_size = 0;
    }    

    double *x_sub = (double *)malloc(local_x_size * sizeof(double));
    if (!x_sub)
    {
        fprintf(stderr, "Rank:%lld Cannot allocate x_sub of size %lld\n", rank, local_x_size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    long long i;
    for (i = 0; i < local_x_size; i++)
    {
        x_sub[i] = (double)(rand() % 1000);
    }

    double *yLocal = (double *)calloc(local_n, sizeof(double));
    if (!yLocal)
    {
        fprintf(stderr, " Failed to allocate yLocal @ Rank: %lld\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Determine neighbors
    long long left_neighbor = (rank == 0) ? MPI_PROC_NULL : rank - 1;         // Left boundary
    long long right_neighbor = (rank == size - 1) ? MPI_PROC_NULL : rank + 1; // Right boundary

    // Allocate buffers for halo regions
    double left_halo[5] = {0};  // Buffer to receive 5 elements from left neighbor
    double right_halo[5] = {0}; // Buffer to receive 5 elements from right neighbor
    double left_send[5] = {0};  // Buffer to send 5 elements to left neighbor
    double right_send[5] = {0}; // Buffer to send 5 elements to right neighbor

    // Fill send buffers with boundary values
    if (local_x_size >= 5)
    {
        int haloSend;
        for (haloSend = 0; haloSend < 5; haloSend++)
        {
            if (offset > 0)
                left_send[haloSend] = x_sub[haloSend]; // First 5 elements for left neighbor
            if (offset + local_n < N)
                right_send[haloSend] = x_sub[local_x_size - 5 + haloSend]; // Last 5 elements for right neighbor
        }
    }

    // Exchange halo data with neighbors
    MPI_Sendrecv(right_send, 5, MPI_DOUBLE, right_neighbor, 0,
                 left_halo, 5, MPI_DOUBLE, left_neighbor, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(left_send, 5, MPI_DOUBLE, left_neighbor, 1,
                 right_halo, 5, MPI_DOUBLE, right_neighbor, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Barrier(MPI_COMM_WORLD);
    // Extend x_sub to include halo elements
    double *x_extended = (double *)malloc((local_x_size + 10) * sizeof(double));
    if (!x_extended)
    {
        fprintf(stderr, "Failed to allocate x_extended at Rank %lld\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Copy values to the extended array
    memcpy(x_extended, left_halo, 5 * sizeof(double));                     // Add left halo
    memcpy(x_extended + 5, x_sub, local_x_size * sizeof(double));          // Add local data
    memcpy(x_extended + 5 + local_x_size, right_halo, 5 * sizeof(double)); // Add right halo

    // SpmvMCsrLocally(local_n, offset, halo_start, local_x_size, row_ptr_local, col_ind_local, val_local, x_sub, yLocal);
    SpmvMCsrLocally(local_n, offset, halo_start, local_x_size + 10, row_ptr_local, col_ind_local, val_local, x_extended, yLocal);

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

        long long offset_r = 0;
        long long rr;
        for (rr = 0; rr < size; rr++)
        {
            long long local_n_r = base + ((rr < rem) ? 1 : 0);
            recvCounts[rr] = (int)local_n_r;
            displs[rr] = (int)offset_r;
            offset_r += local_n_r;
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
            printf("[Rank %lld] recvCounts[%d]: %d, displs[%d]: %d\n", rank, rr, recvCounts[rr], rr, displs[rr]);
            fprintf(fp, "[Rank %lld] recvCounts[%d]: %d, displs[%d]: %d\n", rank, rr, recvCounts[rr], rr, displs[rr]);
        }

        long long totalRows = 0;
        int ii;
        for (ii = 0; ii < size; ii++)
        {
            totalRows += recvCounts[ii];
        }
        if (totalRows != N)
        {
            fprintf(stderr, "[Rank: %lld] Total recvcounts (%lld) != N (%lld)\n", rank, totalRows, N);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int gCode = MPI_Gatherv(yLocal, local_n, MPI_DOUBLE, yGlobal, recvCounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (gCode != MPI_SUCCESS)
    {
        char errMsg[MPI_MAX_ERROR_STRING];
        int msgLen;
        MPI_Error_string(gCode, errMsg, &msgLen);
        fprintf(stderr, "Error in MPI_Gatherv at rank %lld: %s\n", rank, errMsg);
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
        free(yGlobal);
        free(recvCounts);
        free(displs);
        fclose(fp);
    }

    free(yLocal);
    free(x_sub);
    free(row_ptr_local);
    free(col_ind_local);
    free(val_local);

    MPI_Finalize();
    return 0;
}