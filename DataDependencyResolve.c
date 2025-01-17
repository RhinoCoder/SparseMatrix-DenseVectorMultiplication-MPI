#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <papi.h>
#include <limits.h>

/* ---------------------------------------------------------------------
 * GenerateElevenBandedCsrLocal (from your Code1):
 *   Creates a local portion of an 11-banded matrix in CSR format.
 *   Each MPI rank owns [offset..offset+local_n-1] rows and columns
 *   in [halo_start..halo_end].
 *   This version sets all matrix entries to 1.0 instead of random.
 * ---------------------------------------------------------------------*/
void GenerateElevenBandedCsrLocal(long long rank,
                                  long long N,
                                  long long offset,
                                  long long local_n,
                                  long long **row_ptr,
                                  long long **col_ind,
                                  double **val)
{
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

    /* Count the local row bandwidth for each row. */
    long long i_local2;
    for (i_local2 = 0; i_local2 < local_n; i_local2++)
    {
        long long i_global = offset + i_local2;
        long long start_col = (i_global - 5 < 0) ? 0 : (i_global - 5);
        long long end_col = (i_global + 5 >= N) ? (N - 1) : (i_global + 5);
        nnz_per_row[i_local2] = end_col - start_col + 1;
    }

    /* Build row_ptr via prefix sum. */
    (*row_ptr)[0] = 0;
    long long i;
    for (i = 1; i <= local_n; i++)
    {
        (*row_ptr)[i] = (*row_ptr)[i - 1] + nnz_per_row[i - 1];
    }

    /* Figure out how many total nonzeros we have locally. */
    long long local_nnz = 0;
    long long halo_start = (offset - 5 < 0) ? 0 : offset - 5;
    long long halo_end = ((offset + local_n - 1) + 5 >= N) ? (N - 1) : (offset + local_n - 1 + 5);

    if (halo_start < 0 || halo_end >= N)
    {
        fprintf(stderr, "[Rank %lld] Halo region out of bounds: [%lld, %lld]\n", rank, halo_start, halo_end);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* We re-count how many columns actually lie in the [halo_start..halo_end] range. */
    long long i_local;
    for (i_local = 0; i_local < local_n; ++i_local)
    {
        long long start = (*row_ptr)[i_local];
        long long end = (*row_ptr)[i_local + 1];
        long long i_global = offset + i_local;
        /* "k" goes over the band for row i_local, but j_global might be restricted
           to [halo_start..halo_end]. */
        long long k;
        for (k = start; k < end; ++k)
        {
            long long j_global = (i_global - 5 < 0) ? 0
                                                    : (i_global - 5) + (k - start);
            if (j_global >= halo_start && j_global <= halo_end)
            {
                local_nnz++;
            }
        }
    }

    *col_ind = (long long *)malloc(local_nnz * sizeof(long long));
    *val = (double *)malloc(local_nnz * sizeof(double));
    if (!(*col_ind) || !(*val))
    {
        fprintf(stderr, "[Rank %lld] Error: failed to allocate col_ind/val.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Fill col_ind and val with 1.0 within the halo limits. */
    long long idx = 0;
    long long i_local3;
    for (i_local3 = 0; i_local3 < local_n; ++i_local3)
    {
        long long i_global = offset + i_local3;
        long long row_start = (*row_ptr)[i_local3];
        long long row_end = (*row_ptr)[i_local3 + 1];
        long long start_col = (i_global - 5 < 0) ? 0 : (i_global - 5);
        /* Each 'k' goes from row_start..row_end for row i_local3 */
        long long k;
        for (k = row_start; k < row_end; ++k)
        {
            long long j_global = start_col + (k - row_start);
            if (j_global >= halo_start && j_global <= halo_end)
            {
                (*col_ind)[idx] = j_global;
                (*val)[idx] = rand() % 1000;
                idx++;
            }
        }
    }

    free(nnz_per_row);
}

/* ---------------------------------------------------------------------
 * spmv_csr_local (from your Code1):
 *    Multiplies the local banded block by the local slice of x:
 *      y_local[i_local] = sum over (val[k] * x_sub[ j_global - halo_start ])
 * ---------------------------------------------------------------------*/
void spmv_csr_local(long long local_n,
                    long long offset,
                    long long halo_start,
                    long long local_x_size,
                    const long long *row_ptr,
                    const long long *col_ind,
                    const double *val,
                    const double *x_sub,
                    double *y_local)
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
        y_local[i_local] = sum;
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank_int, size_int;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_int);
    MPI_Comm_size(MPI_COMM_WORLD, &size_int);

    long long rank = (long long)rank_int;
    long long size = (long long)size_int;

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

    /* Only rank 0 will write to the output file. */
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

    /* Partition the rows across ranks (block distribution). */
    long long base = N / size;
    long long rem = N % size;
    long long local_n = base + ((rank < rem) ? 1 : 0);

    /* 'offset' is the first row in the global matrix that this rank owns. */
    long long offset = 0;
    long long r;
    for (r = 0; r < rank; r++)
    {
        offset += base + ((r < rem) ? 1 : 0);
    }

    srand((unsigned)time(NULL) + (unsigned)rank * 5843);

    /* Generate local portion of the banded matrix, all 1.0s (Code1 style). */
    long long *row_ptr_local = NULL;
    long long *col_ind_local = NULL;
    double *val_local = NULL;
    GenerateElevenBandedCsrLocal(rank, N, offset, local_n,
                                 &row_ptr_local, &col_ind_local, &val_local);

    /* Determine the halo region so we know how large a portion of x we need. */
    long long halo_start = (offset - 5 < 0) ? 0 : (offset - 5);
    long long halo_end = ((offset + local_n - 1) + 5 >= N) ? (N - 1)
                                                           : (offset + local_n - 1 + 5);
    long long local_x_size = halo_end - halo_start + 1;

    /* Allocate x_sub and fill with random values (similar to Code2). */
    double *x_sub = (double *)malloc(local_x_size * sizeof(double));
    if (!x_sub)
    {
        fprintf(stderr, "[Rank %lld] Error: cannot allocate x_sub of size %lld\n",
                rank, local_x_size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    long long i;
    for (i = 0; i < local_x_size; i++)
    {
        x_sub[i] = (double)(rand() % 1000);
    }

    /* Allocate local result vector y_local. */
    double *y_local = (double *)calloc(local_n, sizeof(double));
    if (!y_local)
    {
        fprintf(stderr, "[Rank %lld] Error: failed to allocate y_local.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Time the local SpMV. */
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    spmv_csr_local(local_n, offset, halo_start, local_x_size,
                   row_ptr_local, col_ind_local, val_local,
                   x_sub, y_local);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double local_ms = (t1 - t0) * 1000.0; /* in milliseconds */

    /* Gather all local results into y_global on rank 0 (like Code2). */
    double *y_global = NULL;
    int *recvcounts = NULL;
    int *displs = NULL;

    if (rank == 0)
    {
        y_global = (double *)malloc(N * sizeof(double));
        recvcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));

        /* Build the gather metadata. */
        long long offset_r = 0;
        long long rr;
        for (rr = 0; rr < size; rr++)
        {
            long long local_n_r = base + ((rr < rem) ? 1 : 0);
            recvcounts[rr] = (int)local_n_r;
            displs[rr] = (int)offset_r;
            offset_r += local_n_r;
        }
    }

    if (rank == 0)
    {
        int rr;
        for (rr = 0; rr < size; rr++)
        {
            printf("[Rank %lld] recvcounts[%d]: %d, displs[%d]: %d\n", rank, rr, recvcounts[rr], rr, displs[rr]);
            fprintf(fp,"[Rank %lld] recvcounts[%d]: %d, displs[%d]: %d\n", rank, rr, recvcounts[rr], rr, displs[rr]);
        }

        long long total_rows = 0;
        for (int i = 0; i < size; i++)
        {
            total_rows += recvcounts[i];
        }
        if (total_rows != N)
        {
            fprintf(stderr, "[Rank %lld] Total recvcounts (%lld) != N (%lld)\n", rank, total_rows, N);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    int gatherv_err = MPI_Gatherv(
        y_local, (int)local_n, MPI_DOUBLE,
        y_global, recvcounts, displs, MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    if (gatherv_err != MPI_SUCCESS)
    {
        char errMsg[MPI_MAX_ERROR_STRING];
        int msgLen;
        MPI_Error_string(gatherv_err, errMsg, &msgLen);
        fprintf(stderr, "Error in MPI_Gatherv at rank %lld: %s\n", rank, errMsg);
        MPI_Abort(MPI_COMM_WORLD, gatherv_err);
    }

    /* Rank 0 can now print partial results from y_global. */
    if (rank == 0)
    {
        fprintf(fp, "> Showing results.....\n\n");
        fprintf(fp, "y_global (first 5 entries): ");

        long long i;
        for (i = 0; i < 5 && i < N; i++)
        {
            fprintf(fp, "%.2f ", y_global[i]);
        }
        fprintf(fp, "\n");
        fprintf(fp, "...............................................\n");
        fprintf(fp, ".....SpMV (Row-block) with N=%lld, Proc Count=%lld .....\n", N, size);
        fprintf(fp, "...............................................\n\n");
        fprintf(fp, "Elapsed time @ [rank 0]: %.3f ms\n\n\n", local_ms);
        fprintf(fp, "...............................................\n");
        fprintf(fp, "PAPI Profiling Results\n");
        fprintf(fp, "...............................................\n\n");
        // If you want to print more data, e.g. PAPI counters, do so here.
        /* Clean up rank 0's gather buffers and file handle. */
        free(y_global);
        free(recvcounts);
        free(displs);
        fclose(fp);
    }

    /* Free local buffers. */
    free(y_local);
    free(x_sub);
    free(row_ptr_local);
    free(col_ind_local);
    free(val_local);

    MPI_Finalize();
    return 0;
}
