!
! ODPSDL_MODEL_IO
!
! The deep learning model computing module includes the pytorch model for loading the C interface, 
! which is used for data standardization and de-standardization functions.
!
!
!

MODULE ODPSDL_MODEL_IO
  USE ISO_C_BINDING
  USE ARMS_Common_Basic


  ! Disable implicit typing
  ! --------------
  IMPLICIT NONE
  REAL(AUS), PARAMETER :: MIN_LayOD = 1.0e-16_AUS 
  REAL(AUS), PARAMETER :: MAX_LayOD = 20.0_AUS
  !Standardizer data structure
  ! --------------
  TYPE Scaler_Type
    REAL(AUS) :: mean        ! layer standardized mean
    REAL(AUS) :: scale       ! layer standardized std
  END TYPE Scaler_Type

  TYPE(Scaler_Type), ALLOCATABLE :: Scaler_X(:,:)    ! Input normalizer (layer, predictor)
  TYPE(Scaler_Type), ALLOCATABLE :: Scaler_Y(:,:)    ! Output normalizer (channel, layer)

  ! C++ model interface
  ! --------------
  INTERFACE
    !Forward interface
    SUBROUTINE run_model_cpp(input_data, output_data, n_profiles, n_layers, n_channels,n_predictors) BIND(C, name="run_model") 
      USE ISO_C_BINDING
      REAL(C_FLOAT), INTENT(IN)  :: input_data(*)
      REAL(C_FLOAT), INTENT(OUT) :: output_data(*)
      INTEGER(C_INT), INTENT(IN), VALUE :: n_profiles, n_layers, n_channels,n_predictors
    END SUBROUTINE run_model_cpp

    !Tangent liner interface
    SUBROUTINE run_model_tangent_linear_cpp( input_data, input_tl_data, output_tl_data, n_profiles, n_layers, n_channels, n_predictors) BIND(C, NAME="run_model_tangent_linear")
      USE ISO_C_BINDING
      REAL(C_FLOAT), INTENT(IN)  :: input_data(*)
      REAL(C_FLOAT), INTENT(IN)  :: input_tl_data(*)
      REAL(C_FLOAT), INTENT(OUT) :: output_tl_data(*)
      INTEGER(C_INT), VALUE :: n_profiles, n_layers, n_channels, n_predictors
    END SUBROUTINE run_model_tangent_linear_cpp

    !Adjoint interface
    SUBROUTINE run_model_adjoint_cpp( input_data, output_adjoint_data, input_adjoint_data, n_profiles, n_layers, n_channels, n_predictors) BIND(C, NAME="run_model_adjoint")
      USE ISO_C_BINDING
      REAL(C_FLOAT), INTENT(IN)  :: input_data(*)
      REAL(C_FLOAT), INTENT(IN)  :: output_adjoint_data(*)
      REAL(C_FLOAT), INTENT(OUT) :: input_adjoint_data(*)
      INTEGER(C_INT), VALUE :: n_profiles, n_layers, n_channels, n_predictors
    END SUBROUTINE run_model_adjoint_cpp

  END INTERFACE
CONTAINS

 ! Call the pytoch model
 ! --------------
 ! Foward model
  SUBROUTINE Run_Model(X_scaled, n_profiles, n_layers, n_channels,n_Predictors, Y_pred_scaled)
    REAL(AUS), INTENT(IN)  :: X_scaled(:,:,:)     ! (n_profiles, n_layers, N_Predictor)
    INTEGER, INTENT(IN)   :: n_profiles, n_layers, n_channels,n_Predictors
    REAL(AUS), INTENT(OUT) :: Y_pred_scaled(:,:,:) ! (n_profiles, n_layers, n_channels)
    
    REAL(C_FLOAT), ALLOCATABLE :: input_c(:), output_c(:)
    INTEGER ::  input_size, output_size
    INTEGER :: i, j, k, idx
    
    input_size = n_profiles * n_layers * n_Predictors
    output_size = n_profiles * n_layers * n_channels

    IF (.NOT. ALLOCATED(input_c)) THEN
      ALLOCATE(input_c(input_size))
    END IF
    IF (.NOT. ALLOCATED(output_c)) THEN
      ALLOCATE(output_c(output_size))
    END IF    
    
    ! Flatten the data into the format required by C++ [batch, layer, features]
    ! --------------
    idx = 1
    DO i = 1, n_profiles
      DO k = 1, n_layers
        DO j = 1, n_Predictors
          input_c(idx) = REAL(X_scaled(i, k, j), C_FLOAT)
          idx = idx + 1
        END DO
      END DO
    END DO
    
    ! Call the C++ model
    ! --------------
    CALL run_model_cpp(input_c, output_c, INT(n_profiles, C_INT), INT(n_layers, C_INT), INT(n_channels, C_INT), INT(n_Predictors, C_INT))
    
    ! Convert the output to a Fortran array [batch, layer, channels]
    ! --------------
    idx = 1
    DO i = 1, n_profiles
      DO k = 1, n_layers
        DO j = 1, n_channels
          Y_pred_scaled(i, k, j) = REAL(output_c(idx), AUS)
          idx = idx + 1
        END DO
      END DO
    END DO
    
    DEALLOCATE(input_c, output_c)
  END SUBROUTINE Run_Model

! TL model
  SUBROUTINE Run_Model_TL( X_scaled, X_scaled_TL, n_profiles, n_layers, n_channels, n_predictors, Y_scaled_TL)
    REAL(AUS), INTENT(IN)  :: X_scaled(:,:,:)     ! [n_profiles, n_layers, n_predictors]
    REAL(AUS), INTENT(IN)  :: X_scaled_TL(:,:,:)  ! [n_profiles, n_layers, n_predictors]
    INTEGER, INTENT(IN)   :: n_profiles, n_layers, n_channels, n_predictors
    REAL(AUS), INTENT(OUT) :: Y_scaled_TL(:,:,:)   ! [n_profiles, n_layers, n_channels]
    
    REAL(C_FLOAT), ALLOCATABLE :: input_c(:), input_tl_c(:), output_tl_c(:)
    INTEGER :: input_size, i, j, k, idx
    
    input_size = n_profiles * n_layers * n_predictors
    ALLOCATE(input_c(input_size), input_tl_c(input_size))
    ALLOCATE(output_tl_c(n_profiles * n_layers * n_channels))
    
    ! Flatten the data [profile, layer, predictor]
    idx = 1
    DO i = 1, n_profiles
      DO j = 1, n_layers
        DO k = 1, n_predictors
          input_c(idx) = REAL(X_scaled(i, j, k), C_FLOAT)
          input_tl_c(idx) = REAL(X_scaled_TL(i, j, k), C_FLOAT)
          idx = idx + 1
        END DO
      END DO
    END DO
    
    ! Call the C++ TL model
    CALL run_model_tangent_linear_cpp( input_c, input_tl_c, output_tl_c, INT(n_profiles, C_INT), INT(n_layers, C_INT), INT(n_channels, C_INT), INT(n_predictors, C_INT))
    
    ! Convert the output to a Fortran array [profile, layer, channel]
    idx = 1
    DO i = 1, n_profiles
      DO j = 1, n_layers
        DO k = 1, n_channels
          Y_scaled_TL(i, j, k) = REAL(output_tl_c(idx), AUS)
          idx = idx + 1
        END DO
      END DO
    END DO
    
    DEALLOCATE(input_c, input_tl_c, output_tl_c)
  END SUBROUTINE Run_Model_TL

! AD model
  SUBROUTINE Run_Model_AD( X_scaled, Y_scaled_AD, n_profiles, n_layers, n_channels, n_predictors, X_scaled_AD)
    REAL(AUS), INTENT(IN)  :: X_scaled(:,:,:)     ! [n_profiles, n_layers, n_predictors]
    REAL(AUS), INTENT(IN)  :: Y_scaled_AD(:,:,:)  ! [n_profiles, n_layers, n_channels]
    INTEGER, INTENT(IN)   :: n_profiles, n_layers, n_channels, n_predictors
    REAL(AUS), INTENT(OUT) :: X_scaled_AD(:,:,:)  ! [n_profiles, n_layers, n_predictors]
    
    REAL(C_FLOAT), ALLOCATABLE :: input_c(:), output_adjoint_c(:), input_adjoint_c(:)
    INTEGER :: input_size, output_size, i, j, k, idx
    
    input_size = n_profiles * n_layers * n_predictors
    output_size = n_profiles * n_layers * n_channels
    ALLOCATE(input_c(input_size), output_adjoint_c(output_size), input_adjoint_c(input_size))
    
    ! Flatten the data [profile, layer, predictor]
    idx = 1
    DO i = 1, n_profiles
      DO j = 1, n_layers
        DO k = 1, n_predictors
          input_c(idx) = REAL(X_scaled(i, j, k), C_FLOAT)
          idx = idx + 1
        END DO
      END DO
    END DO
    idx = 1
    DO i = 1, n_profiles
      DO j = 1, n_layers
        DO k = 1, n_channels
          output_adjoint_c(idx) = REAL(Y_scaled_AD(i, j, k), C_FLOAT)
          idx = idx + 1
        END DO
      END DO
    END DO
    ! Call the C++ AD model
    CALL run_model_adjoint_cpp(input_c, output_adjoint_c, input_adjoint_c, INT(n_profiles, C_INT), INT(n_layers, C_INT), INT(n_channels, C_INT), INT(n_predictors, C_INT))
    
    ! Convert the output to a Fortran array [profile, layer, predictor]
    idx = 1
    DO i = 1, n_profiles
      DO j = 1, n_layers
        DO k = 1, n_predictors
          X_scaled_AD(i, j, k) = REAL(input_adjoint_c(idx), AUS)
          idx = idx + 1
        END DO
      END DO
    END DO
    
    DEALLOCATE(input_c, output_adjoint_c, input_adjoint_c)
  END SUBROUTINE Run_Model_AD

  ! Load standardized parameters
  ! --------------
  SUBROUTINE Load_Scalers(n_layers, n_channels, n_predictors)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: n_layers, n_channels, n_predictors
      INTEGER :: i, j, ierr
      LOGICAL :: file_exists

      ! Allocate memory
      ! --------------
      IF (.NOT. ALLOCATED(Scaler_X)) THEN
        ALLOCATE(Scaler_X(n_layers, n_predictors))
      END IF
      IF (.NOT. ALLOCATED(Scaler_Y)) THEN
        ALLOCATE(Scaler_Y(n_layers, n_channels))
      END IF
      ! Load scaler_X.bin (Each layer and each predictor has its own mean & scale)
      ! --------------
      INQUIRE(FILE='/home/yuan/ARMS_v1.3/coefficients/scaler_X_FY3F.bin', EXIST=file_exists)
      IF (.NOT. file_exists) THEN
          PRINT *, 'File not found: scaler_X.bin'
          STOP
      END IF

      OPEN(UNIT=10, FILE='/home/yuan/ARMS_v1.3/coefficients/scaler_X_FY3F.bin', FORM='UNFORMATTED', ACCESS='STREAM', STATUS='OLD', IOSTAT=ierr)
      IF (ierr /= 0) THEN
          PRINT *, 'Error opening file: scaler_X.bin'
          STOP
      END IF

      ! Read each Scaler_X(i,j) in a loop according to the writing order
      ! --------------
      DO i = 1, n_layers
          DO j = 1, n_predictors
              READ(10) Scaler_X(i,j)%mean, Scaler_X(i,j)%scale
          END DO
      END DO

      CLOSE(10)

      ! Load scaler_Y.bin (Each layer and each output channel has its own mean & scale)
      ! --------------
      INQUIRE(FILE='/home/yuan/ARMS_v1.3/coefficients/scaler_Y_FY3F.bin', EXIST=file_exists)
      IF (.NOT. file_exists) THEN
          PRINT *, 'File not found: scaler_Y.bin'
          STOP
      END IF

      OPEN(UNIT=10, FILE='/home/yuan/ARMS_v1.3/coefficients/scaler_Y_FY3F.bin', FORM='UNFORMATTED', ACCESS='STREAM', STATUS='OLD', IOSTAT=ierr)
      IF (ierr /= 0) THEN
          PRINT *, 'Error opening file: scaler_Y.bin'
          STOP
      END IF

      ! Read each Scaler_Y(j,i) in a loop according to the writing order
      ! --------------
      DO i = 1, n_layers
          DO j = 1, n_channels
              READ(10) Scaler_Y(i,j)%mean, Scaler_Y(i,j)%scale
          END DO
      END DO

      CLOSE(10)


  END SUBROUTINE Load_Scalers

  ! Data preprocessing
  ! --------------
  SUBROUTINE Preprocess_Data(Predictors, n_profiles, n_layers,n_predictors, X_scaled)
    REAL(AUS), INTENT(IN)  :: Predictors(:,:,:) ! (n_profiles, n_layers, n_predictor)
    INTEGER, INTENT(IN)   :: n_profiles, n_layers,n_predictors
    REAL(AUS), INTENT(OUT) :: X_scaled(:,:,:)     ! (n_profiles, n_layers, n_predictor)
    
    INTEGER :: i, j, k
    
    DO i = 1, n_profiles
        DO j = 1, n_layers
          DO k = 1, n_predictors
            ! X_scaled = (x - mean) / scale
            ! --------------
            X_scaled(i, j, k) = (Predictors(i, j, k) - Scaler_X(j,k)%mean) / Scaler_X(j,k)%scale
          END DO
        END DO
    END DO
  END SUBROUTINE Preprocess_Data

  SUBROUTINE Preprocess_Data_TL(Predictors_TL, n_profiles, n_layers,n_predictors, X_scaled_TL)
    REAL(AUS), INTENT(IN)  :: Predictors_TL(:,:,:) ! (n_profiles, n_layers, n_predictor)
    INTEGER, INTENT(IN)   :: n_profiles, n_layers,n_predictors
    REAL(AUS), INTENT(OUT) :: X_scaled_TL(:,:,:)     ! (n_profiles, n_layers, n_predictor)
    
    INTEGER :: i, j, k
    
    DO i = 1, n_profiles
        DO j = 1, n_layers
          DO k = 1, n_predictors
            !  X_scaled = (x - mean) / scale
            !  X_scaled_TL = (x_TL) / scale
            ! --------------
            X_scaled_TL(i, j, k) = Predictors_TL(i, j, k)/ Scaler_X(j,k)%scale
          END DO
        END DO
    END DO
  END SUBROUTINE Preprocess_Data_TL

  SUBROUTINE Preprocess_Data_AD(X_scaled_AD, n_profiles, n_layers,n_predictors, Predictors_AD)
    REAL(AUS), INTENT(IN)  :: X_scaled_AD(:,:,:) ! (n_profiles, n_layers, n_predictor)
    INTEGER, INTENT(IN)   :: n_profiles, n_layers,n_predictors
    REAL(AUS), INTENT(OUT) :: Predictors_AD(:,:,:)     ! (n_profiles, n_layers, n_predictor)
    
    INTEGER :: i, j, k

    DO i = 1, n_profiles
        DO j = 1, n_layers
          DO k = 1, n_predictors
            !  X_scaled = (x - mean) / scale
            !  X_scaled_TL = (x_TL) / scale
            !  Predictors_AD = (X_scaled_AD) / scale
            ! --------------
            Predictors_AD(i, j, k) = Predictors_AD(i, j, k) + X_scaled_AD(i, j, k)/ Scaler_X(j,k)%scale
          END DO
        END DO
    END DO
  END SUBROUTINE Preprocess_Data_AD

  ! Post-processing and anti-standardization
  ! --------------
  SUBROUTINE Postprocess_Output(Y_pred_scaled, n_profiles, n_layers, n_channels, OpticalDepth)
    INTEGER, INTENT(IN)   :: n_profiles, n_layers, n_channels
    REAL(AUS), INTENT(IN)  :: Y_pred_scaled(:,:,:)  ! (n_profiles, n_layers, n_channels)
    REAL(AUS), INTENT(OUT) :: OpticalDepth(:,:,:) ! (n_profiles, n_layers, n_channels)
    
    INTEGER :: i, j, k
    REAL(AUS) :: tmp
    
    ! Reshape it to the original dimension
    ! --------------
    DO i = 1, n_profiles
        DO j = 1, n_layers
          DO k = 1, n_channels
            ! Anti-standardization: y = y_scaled * scale + mean
            ! --------------
            tmp = Y_pred_scaled(i, j, k)
            OpticalDepth(i, j, k) = tmp * Scaler_Y(j,k)%scale + Scaler_Y(j,k)%mean
            
            ! Application constraints
            ! --------------
            IF (OpticalDepth(i, j, k) < MIN_LayOD) THEN
              OpticalDepth(i, j, k) = MIN_LayOD
            elseif (OpticalDepth(i, j, k) > MAX_LayOD) then
              OpticalDepth(i, j, k) = MAX_LayOD
            END IF
           END DO
        END DO
    END DO

  END SUBROUTINE Postprocess_Output
  
  SUBROUTINE Postprocess_Output_TL(Y_scaled_TL, n_profiles, n_layers, n_channels, OpticalDepth_TL)
    INTEGER, INTENT(IN)   :: n_profiles, n_layers, n_channels
    REAL(AUS), INTENT(IN)  :: Y_scaled_TL(:,:,:) ! [n_profiles, n_layers, n_channels]
    REAL(AUS), INTENT(OUT) :: OpticalDepth_TL(:,:,:)! (n_profiles, n_layers, n_channels)
    INTEGER :: i, j, k
    
    ! --------------
    DO i = 1, n_profiles
        DO j = 1, n_layers
          DO k = 1, n_channels
            ! OD = Y_scaled * scale_Y + mean_Y
            ! OD_TL = Y_scaled_TL * scale_Y
            OpticalDepth_TL(i, j, k) = Y_scaled_TL(i, j, k) * Scaler_Y(j,k)%scale 

           END DO
        END DO
    END DO
  END SUBROUTINE Postprocess_Output_TL

  SUBROUTINE Postprocess_Output_AD(OpticalDepth_AD, n_profiles, n_layers, n_channels, Y_scaled_AD)
    INTEGER, INTENT(IN)   :: n_profiles, n_layers, n_channels
    REAL(AUS), INTENT(IN)  :: OpticalDepth_AD(:,:,:) ! [n_profiles, n_layers, n_channels]
    REAL(AUS), INTENT(OUT) :: Y_scaled_AD(:,:,:)! (n_profiles, n_layers, n_channels)
    INTEGER :: i, j, k


    ! --------------
    DO i = 1, n_profiles
        DO j = 1, n_layers
          DO k = 1, n_channels
            ! OD = Y_scaled * scale_Y + mean_Y
            ! OD_TL = Y_scaled_TL * scale_Y
            ! --------------
            Y_scaled_AD(i, j, k) = Y_scaled_AD(i, j, k) + OpticalDepth_AD(i, j, k) * Scaler_Y(j,k)%scale 

           END DO
        END DO
    END DO
  END SUBROUTINE Postprocess_Output_AD

  END MODULE ODPSDL_MODEL_IO