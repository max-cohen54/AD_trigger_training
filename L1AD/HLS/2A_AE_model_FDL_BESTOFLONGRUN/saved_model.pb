��"
��
.
Abs
x"T
y"T"
Ttype:

2	
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%��L>"
Ttype0:
2
:
Less
x"T
y"T
z
"
Ttype:
2	
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
,
Log
x"T
y"T"
Ttype:

2
#
	LogicalOr
x

y

z
�
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
8
Pow
x"T
y"T
z"T"
Ttype:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
2
Round
x"T
y"T"
Ttype:
2
	
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
1
Sign
x"T
y"T"
Ttype:
2
	
-
Sqrt
x"T
y"T"
Ttype:

2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.13.12v2.13.0-17-gf841394b1b78ľ 
n
z_mean/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namez_mean/bias
g
z_mean/bias/Read/ReadVariableOpReadVariableOpz_mean/bias*
_output_shapes
:*
dtype0
v
z_mean/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namez_mean/kernel
o
!z_mean/kernel/Read/ReadVariableOpReadVariableOpz_mean/kernel*
_output_shapes

:*
dtype0
~
BN2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameBN2/moving_variance
w
'BN2/moving_variance/Read/ReadVariableOpReadVariableOpBN2/moving_variance*
_output_shapes
:*
dtype0
v
BN2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameBN2/moving_mean
o
#BN2/moving_mean/Read/ReadVariableOpReadVariableOpBN2/moving_mean*
_output_shapes
:*
dtype0
h
BN2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
BN2/beta
a
BN2/beta/Read/ReadVariableOpReadVariableOpBN2/beta*
_output_shapes
:*
dtype0
j
	BN2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	BN2/gamma
c
BN2/gamma/Read/ReadVariableOpReadVariableOp	BN2/gamma*
_output_shapes
:*
dtype0
n
dense2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense2/bias
g
dense2/bias/Read/ReadVariableOpReadVariableOpdense2/bias*
_output_shapes
:*
dtype0
v
dense2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense2/kernel
o
!dense2/kernel/Read/ReadVariableOpReadVariableOpdense2/kernel*
_output_shapes

: *
dtype0
~
BN1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameBN1/moving_variance
w
'BN1/moving_variance/Read/ReadVariableOpReadVariableOpBN1/moving_variance*
_output_shapes
: *
dtype0
v
BN1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameBN1/moving_mean
o
#BN1/moving_mean/Read/ReadVariableOpReadVariableOpBN1/moving_mean*
_output_shapes
: *
dtype0
h
BN1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
BN1/beta
a
BN1/beta/Read/ReadVariableOpReadVariableOpBN1/beta*
_output_shapes
: *
dtype0
j
	BN1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	BN1/gamma
c
BN1/gamma/Read/ReadVariableOpReadVariableOp	BN1/gamma*
_output_shapes
: *
dtype0
n
dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense1/bias
g
dense1/bias/Read/ReadVariableOpReadVariableOpdense1/bias*
_output_shapes
: *
dtype0
v
dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:, *
shared_namedense1/kernel
o
!dense1/kernel/Read/ReadVariableOpReadVariableOpdense1/kernel*
_output_shapes

:, *
dtype0
y
serving_default_inputsPlaceholder*'
_output_shapes
:���������,*
dtype0*
shape:���������,
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputsdense1/kerneldense1/bias	BN1/gammaBN1/betaBN1/moving_meanBN1/moving_variancedense2/kerneldense2/bias	BN2/gammaBN2/betaBN2/moving_meanBN2/moving_variancez_mean/kernelz_mean/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_8205179

NoOpNoOp
�:
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�9
value�9B�9 B�9
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
kernel_quantizer
bias_quantizer
kernel_quantizer_internal
bias_quantizer_internal

quantizers

kernel
bias*
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"beta_quantizer_internal
#gamma_quantizer_internal
$mean_quantizer_internal
%variance_quantizer_internal
&
quantizers
'axis
	(gamma
)beta
*moving_mean
+moving_variance*
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses
2
activation
2	quantizer* 
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses
9kernel_quantizer
:bias_quantizer
9kernel_quantizer_internal
:bias_quantizer_internal
;
quantizers

<kernel
=bias*
�
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses
Dbeta_quantizer_internal
Egamma_quantizer_internal
Fmean_quantizer_internal
Gvariance_quantizer_internal
H
quantizers
Iaxis
	Jgamma
Kbeta
Lmoving_mean
Mmoving_variance*
�
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses
T
activation
T	quantizer* 
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses
[kernel_quantizer
\bias_quantizer
[kernel_quantizer_internal
\bias_quantizer_internal
]
quantizers

^kernel
_bias*
j
0
1
(2
)3
*4
+5
<6
=7
J8
K9
L10
M11
^12
_13*
J
0
1
(2
)3
<4
=5
J6
K7
^8
_9*
* 
�
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

etrace_0
ftrace_1* 

gtrace_0
htrace_1* 
* 

iserving_default* 

0
1*

0
1*
* 
�
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

otrace_0* 

ptrace_0* 
* 
* 

0
1* 
]W
VARIABLE_VALUEdense1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEdense1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
(0
)1
*2
+3*

(0
)1*
* 
�
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*

vtrace_0
wtrace_1* 

xtrace_0
ytrace_1* 
* 
* 
* 
* 

#0
"1
$2
%3* 
* 
XR
VARIABLE_VALUE	BN1/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEBN1/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEBN1/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEBN1/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses* 

trace_0* 

�trace_0* 
* 

<0
=1*

<0
=1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 

90
:1* 
]W
VARIABLE_VALUEdense2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEdense2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
J0
K1
L2
M3*

J0
K1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 

E0
D1
F2
G3* 
* 
XR
VARIABLE_VALUE	BN2/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEBN2/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEBN2/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEBN2/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 

^0
_1*

^0
_1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 

[0
\1* 
]W
VARIABLE_VALUEz_mean/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEz_mean/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
*0
+1
L2
M3*
<
0
1
2
3
4
5
6
7*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

*0
+1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

L0
M1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense1/kerneldense1/bias	BN1/gammaBN1/betaBN1/moving_meanBN1/moving_variancedense2/kerneldense2/bias	BN2/gammaBN2/betaBN2/moving_meanBN2/moving_variancez_mean/kernelz_mean/biasConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_save_8207142
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense1/kerneldense1/bias	BN1/gammaBN1/betaBN1/moving_meanBN1/moving_variancedense2/kerneldense2/bias	BN2/gammaBN2/betaBN2/moving_meanBN2/moving_variancez_mean/kernelz_mean/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__traced_restore_8207193��
��
�
@__inference_BN2_layer_call_and_return_conditional_losses_8204217

inputs%
readvariableop_resource:'
readvariableop_3_resource:'
readvariableop_6_resource:,
relu_3_readvariableop_resource:
identity��Abs_1/ReadVariableOp�Abs_3/ReadVariableOp�AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_10�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�ReadVariableOp_6�ReadVariableOp_7�ReadVariableOp_8�ReadVariableOp_9�Relu/ReadVariableOp�Relu_1/ReadVariableOp�Relu_3/ReadVariableOp�Relu_4/ReadVariableOpb
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0P
LessEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ei
	LessEqual	LessEqualReadVariableOp:value:0LessEqual/y:output:0*
T0*
_output_shapes
:g
Relu/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0N
ReluReluRelu/ReadVariableOp:value:0*
T0*
_output_shapes
:l
ones_like/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0i
ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?z
	ones_likeFill(ones_like/Shape/shape_as_tensor:output:0ones_like/Const:output:0*
T0*
_output_shapes
:J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ES
mulMulones_like:output:0mul/y:output:0*
T0*
_output_shapes
:e
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*
_output_shapes
:i
Relu_1/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0R
Relu_1ReluRelu_1/ReadVariableOp:value:0*
T0*
_output_shapes
:K
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3X
LessLessRelu_1:activations:0Less/y:output:0*
T0*
_output_shapes
:Q
SelectV2_1/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3r

SelectV2_1SelectV2Less:z:0SelectV2_1/t:output:0Relu_1:activations:0*
T0*
_output_shapes
:S
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Eo
GreaterEqualGreaterEqualSelectV2_1:output:0GreaterEqual/y:output:0*
T0*
_output_shapes
:k
!ones_like_1/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_1Fill*ones_like_1/Shape/shape_as_tensor:output:0ones_like_1/Const:output:0*
T0*
_output_shapes
:L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   EY
mul_1Mulones_like_1:output:0mul_1/y:output:0*
T0*
_output_shapes
:m

SelectV2_2SelectV2GreaterEqual:z:0	mul_1:z:0SelectV2_1:output:0*
T0*
_output_shapes
:M
Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3\
Less_1LessRelu_1:activations:0Less_1/y:output:0*
T0*
_output_shapes
:k
!ones_like_2/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_2Fill*ones_like_2/Shape/shape_as_tensor:output:0ones_like_2/Const:output:0*
T0*
_output_shapes
:L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �Y
mul_2Mulones_like_2:output:0mul_2/y:output:0*
T0*
_output_shapes
:D
LogLogSelectV2_2:output:0*
T0*
_output_shapes
:N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?T
truedivRealDivLog:y:0truediv/y:output:0*
T0*
_output_shapes
:<
NegNegtruediv:z:0*
T0*
_output_shapes
:@
RoundRoundtruediv:z:0*
T0*
_output_shapes
:E
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes
:J
StopGradientStopGradientadd:z:0*
T0*
_output_shapes
:W
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes
:\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Ar
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �r
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes
:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_3Mulmul_3/x:output:0clip_by_value:z:0*
T0*
_output_shapes
:]

SelectV2_3SelectV2
Less_1:z:0	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:d
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0K
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes
:>
Relu_2Relu	Neg_1:y:0*
T0*
_output_shapes
:L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
mul_4MulRelu_2:activations:0mul_4/y:output:0*
T0*
_output_shapes
:M
Less_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_2Less	mul_4:z:0Less_2/y:output:0*
T0*
_output_shapes
:Q
SelectV2_4/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3i

SelectV2_4SelectV2
Less_2:z:0SelectV2_4/t:output:0	mul_4:z:0*
T0*
_output_shapes
:U
GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Es
GreaterEqual_1GreaterEqualSelectV2_4:output:0GreaterEqual_1/y:output:0*
T0*
_output_shapes
:k
!ones_like_3/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_3Fill*ones_like_3/Shape/shape_as_tensor:output:0ones_like_3/Const:output:0*
T0*
_output_shapes
:L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *   EY
mul_5Mulones_like_3:output:0mul_5/y:output:0*
T0*
_output_shapes
:o

SelectV2_5SelectV2GreaterEqual_1:z:0	mul_5:z:0SelectV2_4:output:0*
T0*
_output_shapes
:M
Less_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_3Less	mul_4:z:0Less_3/y:output:0*
T0*
_output_shapes
:k
!ones_like_4/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_4Fill*ones_like_4/Shape/shape_as_tensor:output:0ones_like_4/Const:output:0*
T0*
_output_shapes
:L
mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �Y
mul_6Mulones_like_4:output:0mul_6/y:output:0*
T0*
_output_shapes
:F
Log_1LogSelectV2_5:output:0*
T0*
_output_shapes
:P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_1RealDiv	Log_1:y:0truediv_1/y:output:0*
T0*
_output_shapes
:@
Neg_2Negtruediv_1:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_1:z:0*
T0*
_output_shapes
:K
add_2AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_1StopGradient	add_2:z:0*
T0*
_output_shapes
:[
add_3AddV2truediv_1:z:0StopGradient_1:output:0*
T0*
_output_shapes
:^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Av
clip_by_value_1/MinimumMinimum	add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?X
mul_7Mulmul_7/x:output:0clip_by_value_1:z:0*
T0*
_output_shapes
:]

SelectV2_6SelectV2
Less_3:z:0	mul_6:z:0	mul_7:z:0*
T0*
_output_shapes
:d
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0U
GreaterEqual_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    x
GreaterEqual_2GreaterEqualReadVariableOp_2:value:0GreaterEqual_2/y:output:0*
T0*
_output_shapes
:M
LogicalOr/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z\
	LogicalOr	LogicalOrGreaterEqual_2:z:0LogicalOr/y:output:0*
_output_shapes
:J
pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
powPowpow/x:output:0SelectV2_3:output:0*
T0*
_output_shapes
:L
pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @X
pow_1Powpow_1/x:output:0SelectV2_6:output:0*
T0*
_output_shapes
:<
Neg_3Neg	pow_1:z:0*
T0*
_output_shapes
:^

SelectV2_7SelectV2LogicalOr:z:0pow:z:0	Neg_3:y:0*
T0*
_output_shapes
:D
Neg_4NegSelectV2:output:0*
T0*
_output_shapes
:S
add_4AddV2	Neg_4:y:0SelectV2_7:output:0*
T0*
_output_shapes
:L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_8Mulmul_8/x:output:0	add_4:z:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	mul_8:z:0*
T0*
_output_shapes
:_
add_5AddV2SelectV2:output:0StopGradient_2:output:0*
T0*
_output_shapes
:f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0K
SignSignReadVariableOp_3:value:0*
T0*
_output_shapes
:9
AbsAbsSign:y:0*
T0*
_output_shapes
:J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?H
subSubsub/x:output:0Abs:y:0*
T0*
_output_shapes
:F
add_6AddV2Sign:y:0sub:z:0*
T0*
_output_shapes
:j
Abs_1/ReadVariableOpReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0O
Abs_1AbsAbs_1/ReadVariableOp:value:0*
T0*
_output_shapes
:M
Less_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_4Less	Abs_1:y:0Less_4/y:output:0*
T0*
_output_shapes
:Q
SelectV2_8/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3i

SelectV2_8SelectV2
Less_4:z:0SelectV2_8/t:output:0	Abs_1:y:0*
T0*
_output_shapes
:M
Less_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_5Less	Abs_1:y:0Less_5/y:output:0*
T0*
_output_shapes
:k
!ones_like_5/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_5Fill*ones_like_5/Shape/shape_as_tensor:output:0ones_like_5/Const:output:0*
T0*
_output_shapes
:L
mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �Y
mul_9Mulones_like_5:output:0mul_9/y:output:0*
T0*
_output_shapes
:F
Log_2LogSelectV2_8:output:0*
T0*
_output_shapes
:P
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_2RealDiv	Log_2:y:0truediv_2/y:output:0*
T0*
_output_shapes
:@
Neg_5Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_2Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_7AddV2	Neg_5:y:0Round_2:y:0*
T0*
_output_shapes
:N
StopGradient_3StopGradient	add_7:z:0*
T0*
_output_shapes
:[
add_8AddV2truediv_2:z:0StopGradient_3:output:0*
T0*
_output_shapes
:^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@v
clip_by_value_2/MinimumMinimum	add_8:z:0"clip_by_value_2/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*
_output_shapes
:M
mul_10/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
mul_10Mulmul_10/x:output:0clip_by_value_2:z:0*
T0*
_output_shapes
:^

SelectV2_9SelectV2
Less_5:z:0	mul_9:z:0
mul_10:z:0*
T0*
_output_shapes
:L
pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @X
pow_2Powpow_2/x:output:0SelectV2_9:output:0*
T0*
_output_shapes
:H
mul_11Mul	add_6:z:0	pow_2:z:0*
T0*
_output_shapes
:f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0K
Neg_6NegReadVariableOp_4:value:0*
T0*
_output_shapes
:J
add_9AddV2	Neg_6:y:0
mul_11:z:0*
T0*
_output_shapes
:M
mul_12/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?P
mul_12Mulmul_12/x:output:0	add_9:z:0*
T0*
_output_shapes
:O
StopGradient_4StopGradient
mul_12:z:0*
T0*
_output_shapes
:f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0g
add_10AddV2ReadVariableOp_5:value:0StopGradient_4:output:0*
T0*
_output_shapes
:f
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource*
_output_shapes
:*
dtype0M
Sign_1SignReadVariableOp_6:value:0*
T0*
_output_shapes
:=
Abs_2Abs
Sign_1:y:0*
T0*
_output_shapes
:L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_1Subsub_1/x:output:0	Abs_2:y:0*
T0*
_output_shapes
:K
add_11AddV2
Sign_1:y:0	sub_1:z:0*
T0*
_output_shapes
:j
Abs_3/ReadVariableOpReadVariableOpreadvariableop_6_resource*
_output_shapes
:*
dtype0O
Abs_3AbsAbs_3/ReadVariableOp:value:0*
T0*
_output_shapes
:M
Less_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_6Less	Abs_3:y:0Less_6/y:output:0*
T0*
_output_shapes
:R
SelectV2_10/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3k
SelectV2_10SelectV2
Less_6:z:0SelectV2_10/t:output:0	Abs_3:y:0*
T0*
_output_shapes
:M
Less_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_7Less	Abs_3:y:0Less_7/y:output:0*
T0*
_output_shapes
:k
!ones_like_6/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_6Fill*ones_like_6/Shape/shape_as_tensor:output:0ones_like_6/Const:output:0*
T0*
_output_shapes
:M
mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �[
mul_13Mulones_like_6:output:0mul_13/y:output:0*
T0*
_output_shapes
:G
Log_3LogSelectV2_10:output:0*
T0*
_output_shapes
:P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_3RealDiv	Log_3:y:0truediv_3/y:output:0*
T0*
_output_shapes
:@
Neg_7Negtruediv_3:z:0*
T0*
_output_shapes
:D
Round_3Roundtruediv_3:z:0*
T0*
_output_shapes
:L
add_12AddV2	Neg_7:y:0Round_3:y:0*
T0*
_output_shapes
:O
StopGradient_5StopGradient
add_12:z:0*
T0*
_output_shapes
:\
add_13AddV2truediv_3:z:0StopGradient_5:output:0*
T0*
_output_shapes
:^
clip_by_value_3/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@w
clip_by_value_3/MinimumMinimum
add_13:z:0"clip_by_value_3/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0*
_output_shapes
:M
mul_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
mul_14Mulmul_14/x:output:0clip_by_value_3:z:0*
T0*
_output_shapes
:`
SelectV2_11SelectV2
Less_7:z:0
mul_13:z:0
mul_14:z:0*
T0*
_output_shapes
:L
pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_3Powpow_3/x:output:0SelectV2_11:output:0*
T0*
_output_shapes
:I
mul_15Mul
add_11:z:0	pow_3:z:0*
T0*
_output_shapes
:f
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource*
_output_shapes
:*
dtype0K
Neg_8NegReadVariableOp_7:value:0*
T0*
_output_shapes
:K
add_14AddV2	Neg_8:y:0
mul_15:z:0*
T0*
_output_shapes
:M
mul_16/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
mul_16Mulmul_16/x:output:0
add_14:z:0*
T0*
_output_shapes
:O
StopGradient_6StopGradient
mul_16:z:0*
T0*
_output_shapes
:f
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource*
_output_shapes
:*
dtype0g
add_15AddV2ReadVariableOp_8:value:0StopGradient_6:output:0*
T0*
_output_shapes
:p
Relu_3/ReadVariableOpReadVariableOprelu_3_readvariableop_resource*
_output_shapes
:*
dtype0R
Relu_3ReluRelu_3/ReadVariableOp:value:0*
T0*
_output_shapes
:p
Relu_4/ReadVariableOpReadVariableOprelu_3_readvariableop_resource*
_output_shapes
:*
dtype0R
Relu_4ReluRelu_4/ReadVariableOp:value:0*
T0*
_output_shapes
:M
Less_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3\
Less_8LessRelu_4:activations:0Less_8/y:output:0*
T0*
_output_shapes
:R
SelectV2_12/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3v
SelectV2_12SelectV2
Less_8:z:0SelectV2_12/t:output:0Relu_4:activations:0*
T0*
_output_shapes
:M
Less_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3\
Less_9LessRelu_4:activations:0Less_9/y:output:0*
T0*
_output_shapes
:k
!ones_like_7/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_7Fill*ones_like_7/Shape/shape_as_tensor:output:0ones_like_7/Const:output:0*
T0*
_output_shapes
:M
mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �[
mul_17Mulones_like_7:output:0mul_17/y:output:0*
T0*
_output_shapes
:G
SqrtSqrtSelectV2_12:output:0*
T0*
_output_shapes
:;
Log_4LogSqrt:y:0*
T0*
_output_shapes
:P
truediv_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_4RealDiv	Log_4:y:0truediv_4/y:output:0*
T0*
_output_shapes
:@
Neg_9Negtruediv_4:z:0*
T0*
_output_shapes
:D
Round_4Roundtruediv_4:z:0*
T0*
_output_shapes
:L
add_16AddV2	Neg_9:y:0Round_4:y:0*
T0*
_output_shapes
:O
StopGradient_7StopGradient
add_16:z:0*
T0*
_output_shapes
:\
add_17AddV2truediv_4:z:0StopGradient_7:output:0*
T0*
_output_shapes
:^
clip_by_value_4/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Aw
clip_by_value_4/MinimumMinimum
add_17:z:0"clip_by_value_4/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_4Maximumclip_by_value_4/Minimum:z:0clip_by_value_4/y:output:0*
T0*
_output_shapes
:M
mul_18/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Z
mul_18Mulmul_18/x:output:0clip_by_value_4:z:0*
T0*
_output_shapes
:`
SelectV2_13SelectV2
Less_9:z:0
mul_17:z:0
mul_18:z:0*
T0*
_output_shapes
:k
ReadVariableOp_9ReadVariableOprelu_3_readvariableop_resource*
_output_shapes
:*
dtype0L
Neg_10NegReadVariableOp_9:value:0*
T0*
_output_shapes
:?
Relu_5Relu
Neg_10:y:0*
T0*
_output_shapes
:M
mul_19/yConst*
_output_shapes
: *
dtype0*
valueB
 *    [
mul_19MulRelu_5:activations:0mul_19/y:output:0*
T0*
_output_shapes
:N
	Less_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3T
Less_10Less
mul_19:z:0Less_10/y:output:0*
T0*
_output_shapes
:R
SelectV2_14/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3m
SelectV2_14SelectV2Less_10:z:0SelectV2_14/t:output:0
mul_19:z:0*
T0*
_output_shapes
:N
	Less_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3T
Less_11Less
mul_19:z:0Less_11/y:output:0*
T0*
_output_shapes
:k
!ones_like_8/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_8Fill*ones_like_8/Shape/shape_as_tensor:output:0ones_like_8/Const:output:0*
T0*
_output_shapes
:M
mul_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �[
mul_20Mulones_like_8:output:0mul_20/y:output:0*
T0*
_output_shapes
:I
Sqrt_1SqrtSelectV2_14:output:0*
T0*
_output_shapes
:=
Log_5Log
Sqrt_1:y:0*
T0*
_output_shapes
:P
truediv_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_5RealDiv	Log_5:y:0truediv_5/y:output:0*
T0*
_output_shapes
:A
Neg_11Negtruediv_5:z:0*
T0*
_output_shapes
:D
Round_5Roundtruediv_5:z:0*
T0*
_output_shapes
:M
add_18AddV2
Neg_11:y:0Round_5:y:0*
T0*
_output_shapes
:O
StopGradient_8StopGradient
add_18:z:0*
T0*
_output_shapes
:\
add_19AddV2truediv_5:z:0StopGradient_8:output:0*
T0*
_output_shapes
:^
clip_by_value_5/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Aw
clip_by_value_5/MinimumMinimum
add_19:z:0"clip_by_value_5/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_5Maximumclip_by_value_5/Minimum:z:0clip_by_value_5/y:output:0*
T0*
_output_shapes
:M
mul_21/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Z
mul_21Mulmul_21/x:output:0clip_by_value_5:z:0*
T0*
_output_shapes
:a
SelectV2_15SelectV2Less_11:z:0
mul_20:z:0
mul_21:z:0*
T0*
_output_shapes
:l
ReadVariableOp_10ReadVariableOprelu_3_readvariableop_resource*
_output_shapes
:*
dtype0U
GreaterEqual_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *    y
GreaterEqual_3GreaterEqualReadVariableOp_10:value:0GreaterEqual_3/y:output:0*
T0*
_output_shapes
:O
LogicalOr_1/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z`
LogicalOr_1	LogicalOrGreaterEqual_3:z:0LogicalOr_1/y:output:0*
_output_shapes
:L
pow_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_4Powpow_4/x:output:0SelectV2_13:output:0*
T0*
_output_shapes
:L
pow_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_5Powpow_5/x:output:0SelectV2_15:output:0*
T0*
_output_shapes
:=
Neg_12Neg	pow_5:z:0*
T0*
_output_shapes
:d
SelectV2_16SelectV2LogicalOr_1:z:0	pow_4:z:0
Neg_12:y:0*
T0*
_output_shapes
:H
Neg_13NegRelu_3:activations:0*
T0*
_output_shapes
:V
add_20AddV2
Neg_13:y:0SelectV2_16:output:0*
T0*
_output_shapes
:M
mul_22/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
mul_22Mulmul_22/x:output:0
add_20:z:0*
T0*
_output_shapes
:O
StopGradient_9StopGradient
mul_22:z:0*
T0*
_output_shapes
:c
add_21AddV2Relu_3:activations:0StopGradient_9:output:0*
T0*
_output_shapes
:h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 M
Sign_2Signmoments/Squeeze:output:0*
T0*
_output_shapes
:=
Abs_4Abs
Sign_2:y:0*
T0*
_output_shapes
:L
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_2Subsub_2/x:output:0	Abs_4:y:0*
T0*
_output_shapes
:K
add_22AddV2
Sign_2:y:0	sub_2:z:0*
T0*
_output_shapes
:K
Abs_5Absmoments/Squeeze:output:0*
T0*
_output_shapes
:N
	Less_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3S
Less_12Less	Abs_5:y:0Less_12/y:output:0*
T0*
_output_shapes
:R
SelectV2_17/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3l
SelectV2_17SelectV2Less_12:z:0SelectV2_17/t:output:0	Abs_5:y:0*
T0*
_output_shapes
:N
	Less_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3S
Less_13Less	Abs_5:y:0Less_13/y:output:0*
T0*
_output_shapes
:k
!ones_like_9/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_9/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_9Fill*ones_like_9/Shape/shape_as_tensor:output:0ones_like_9/Const:output:0*
T0*
_output_shapes
:M
mul_23/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �[
mul_23Mulones_like_9:output:0mul_23/y:output:0*
T0*
_output_shapes
:G
Log_6LogSelectV2_17:output:0*
T0*
_output_shapes
:P
truediv_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_6RealDiv	Log_6:y:0truediv_6/y:output:0*
T0*
_output_shapes
:A
Neg_14Negtruediv_6:z:0*
T0*
_output_shapes
:D
Round_6Roundtruediv_6:z:0*
T0*
_output_shapes
:M
add_23AddV2
Neg_14:y:0Round_6:y:0*
T0*
_output_shapes
:P
StopGradient_10StopGradient
add_23:z:0*
T0*
_output_shapes
:]
add_24AddV2truediv_6:z:0StopGradient_10:output:0*
T0*
_output_shapes
:^
clip_by_value_6/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@w
clip_by_value_6/MinimumMinimum
add_24:z:0"clip_by_value_6/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_6Maximumclip_by_value_6/Minimum:z:0clip_by_value_6/y:output:0*
T0*
_output_shapes
:M
mul_24/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
mul_24Mulmul_24/x:output:0clip_by_value_6:z:0*
T0*
_output_shapes
:a
SelectV2_18SelectV2Less_13:z:0
mul_23:z:0
mul_24:z:0*
T0*
_output_shapes
:L
pow_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_6Powpow_6/x:output:0SelectV2_18:output:0*
T0*
_output_shapes
:I
mul_25Mul
add_22:z:0	pow_6:z:0*
T0*
_output_shapes
:L
Neg_15Negmoments/Squeeze:output:0*
T0*
_output_shapes
:L
add_25AddV2
Neg_15:y:0
mul_25:z:0*
T0*
_output_shapes
:M
mul_26/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
mul_26Mulmul_26/x:output:0
add_25:z:0*
T0*
_output_shapes
:P
StopGradient_11StopGradient
mul_26:z:0*
T0*
_output_shapes
:h
add_26AddV2moments/Squeeze:output:0StopGradient_11:output:0*
T0*
_output_shapes
:O
Relu_6Relumoments/Squeeze_1:output:0*
T0*
_output_shapes
:O
Relu_7Relumoments/Squeeze_1:output:0*
T0*
_output_shapes
:N
	Less_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3^
Less_14LessRelu_7:activations:0Less_14/y:output:0*
T0*
_output_shapes
:R
SelectV2_19/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3w
SelectV2_19SelectV2Less_14:z:0SelectV2_19/t:output:0Relu_7:activations:0*
T0*
_output_shapes
:N
	Less_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3^
Less_15LessRelu_7:activations:0Less_15/y:output:0*
T0*
_output_shapes
:l
"ones_like_10/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:W
ones_like_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_10Fill+ones_like_10/Shape/shape_as_tensor:output:0ones_like_10/Const:output:0*
T0*
_output_shapes
:M
mul_27/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �\
mul_27Mulones_like_10:output:0mul_27/y:output:0*
T0*
_output_shapes
:I
Sqrt_2SqrtSelectV2_19:output:0*
T0*
_output_shapes
:=
Log_7Log
Sqrt_2:y:0*
T0*
_output_shapes
:P
truediv_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_7RealDiv	Log_7:y:0truediv_7/y:output:0*
T0*
_output_shapes
:A
Neg_16Negtruediv_7:z:0*
T0*
_output_shapes
:D
Round_7Roundtruediv_7:z:0*
T0*
_output_shapes
:M
add_27AddV2
Neg_16:y:0Round_7:y:0*
T0*
_output_shapes
:P
StopGradient_12StopGradient
add_27:z:0*
T0*
_output_shapes
:]
add_28AddV2truediv_7:z:0StopGradient_12:output:0*
T0*
_output_shapes
:^
clip_by_value_7/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Aw
clip_by_value_7/MinimumMinimum
add_28:z:0"clip_by_value_7/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_7Maximumclip_by_value_7/Minimum:z:0clip_by_value_7/y:output:0*
T0*
_output_shapes
:M
mul_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Z
mul_28Mulmul_28/x:output:0clip_by_value_7:z:0*
T0*
_output_shapes
:a
SelectV2_20SelectV2Less_15:z:0
mul_27:z:0
mul_28:z:0*
T0*
_output_shapes
:N
Neg_17Negmoments/Squeeze_1:output:0*
T0*
_output_shapes
:?
Relu_8Relu
Neg_17:y:0*
T0*
_output_shapes
:M
mul_29/yConst*
_output_shapes
: *
dtype0*
valueB
 *    [
mul_29MulRelu_8:activations:0mul_29/y:output:0*
T0*
_output_shapes
:N
	Less_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3T
Less_16Less
mul_29:z:0Less_16/y:output:0*
T0*
_output_shapes
:R
SelectV2_21/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3m
SelectV2_21SelectV2Less_16:z:0SelectV2_21/t:output:0
mul_29:z:0*
T0*
_output_shapes
:N
	Less_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3T
Less_17Less
mul_29:z:0Less_17/y:output:0*
T0*
_output_shapes
:l
"ones_like_11/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:W
ones_like_11/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_11Fill+ones_like_11/Shape/shape_as_tensor:output:0ones_like_11/Const:output:0*
T0*
_output_shapes
:M
mul_30/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �\
mul_30Mulones_like_11:output:0mul_30/y:output:0*
T0*
_output_shapes
:I
Sqrt_3SqrtSelectV2_21:output:0*
T0*
_output_shapes
:=
Log_8Log
Sqrt_3:y:0*
T0*
_output_shapes
:P
truediv_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_8RealDiv	Log_8:y:0truediv_8/y:output:0*
T0*
_output_shapes
:A
Neg_18Negtruediv_8:z:0*
T0*
_output_shapes
:D
Round_8Roundtruediv_8:z:0*
T0*
_output_shapes
:M
add_29AddV2
Neg_18:y:0Round_8:y:0*
T0*
_output_shapes
:P
StopGradient_13StopGradient
add_29:z:0*
T0*
_output_shapes
:]
add_30AddV2truediv_8:z:0StopGradient_13:output:0*
T0*
_output_shapes
:^
clip_by_value_8/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Aw
clip_by_value_8/MinimumMinimum
add_30:z:0"clip_by_value_8/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_8Maximumclip_by_value_8/Minimum:z:0clip_by_value_8/y:output:0*
T0*
_output_shapes
:M
mul_31/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Z
mul_31Mulmul_31/x:output:0clip_by_value_8:z:0*
T0*
_output_shapes
:a
SelectV2_22SelectV2Less_17:z:0
mul_30:z:0
mul_31:z:0*
T0*
_output_shapes
:U
GreaterEqual_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *    z
GreaterEqual_4GreaterEqualmoments/Squeeze_1:output:0GreaterEqual_4/y:output:0*
T0*
_output_shapes
:O
LogicalOr_2/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z`
LogicalOr_2	LogicalOrGreaterEqual_4:z:0LogicalOr_2/y:output:0*
_output_shapes
:L
pow_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_7Powpow_7/x:output:0SelectV2_20:output:0*
T0*
_output_shapes
:L
pow_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_8Powpow_8/x:output:0SelectV2_22:output:0*
T0*
_output_shapes
:=
Neg_19Neg	pow_8:z:0*
T0*
_output_shapes
:d
SelectV2_23SelectV2LogicalOr_2:z:0	pow_7:z:0
Neg_19:y:0*
T0*
_output_shapes
:H
Neg_20NegRelu_6:activations:0*
T0*
_output_shapes
:V
add_31AddV2
Neg_20:y:0SelectV2_23:output:0*
T0*
_output_shapes
:M
mul_32/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
mul_32Mulmul_32/x:output:0
add_31:z:0*
T0*
_output_shapes
:P
StopGradient_14StopGradient
mul_32:z:0*
T0*
_output_shapes
:d
add_32AddV2Relu_6:activations:0StopGradient_14:output:0*
T0*
_output_shapes
:Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<t
AssignMovingAvg/ReadVariableOpReadVariableOpreadvariableop_6_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOpreadvariableop_6_resourceAssignMovingAvg/mul:z:0^Abs_3/ReadVariableOp^AssignMovingAvg/ReadVariableOp^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<{
 AssignMovingAvg_1/ReadVariableOpReadVariableOprelu_3_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOprelu_3_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp_10^ReadVariableOp_9^Relu_3/ReadVariableOp^Relu_4/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:a
batchnorm/addAddV2
add_32:z:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Y
batchnorm/mulMulbatchnorm/Rsqrt:y:0	add_5:z:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������Z
batchnorm/mul_2Mul
add_26:z:0batchnorm/mul:z:0*
T0*
_output_shapes
:Z
batchnorm/subSub
add_10:z:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^Abs_1/ReadVariableOp^Abs_3/ReadVariableOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_10^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8^ReadVariableOp_9^Relu/ReadVariableOp^Relu_1/ReadVariableOp^Relu_3/ReadVariableOp^Relu_4/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2,
Abs_1/ReadVariableOpAbs_1/ReadVariableOp2,
Abs_3/ReadVariableOpAbs_3/ReadVariableOp2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12&
ReadVariableOp_10ReadVariableOp_102$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82$
ReadVariableOp_9ReadVariableOp_92*
Relu/ReadVariableOpRelu/ReadVariableOp2.
Relu_1/ReadVariableOpRelu_1/ReadVariableOp2.
Relu_3/ReadVariableOpRelu_3/ReadVariableOp2.
Relu_4/ReadVariableOpRelu_4/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
(__inference_z_mean_layer_call_fn_8206968

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_z_mean_layer_call_and_return_conditional_losses_8204364o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:'#
!
_user_specified_name	8206962:'#
!
_user_specified_name	8206964
�/
^
B__inference_relu2_layer_call_and_return_conditional_losses_8206959

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B : Q
Cast_2CastCast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `AG
subSub
Cast_2:y:0sub/y:output:0*
T0*
_output_shapes
: F
Pow_2PowConst:output:0sub:z:0*
T0*
_output_shapes
: D
sub_1Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
	LessEqual	LessEqualinputs	sub_1:z:0*
T0*'
_output_shapes
:���������W
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:���������*
alpha%  �:S
ones_like/ShapeShapeinputs*
T0*
_output_shapes
::��T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:���������D
sub_2Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
mulMulones_like:output:0	sub_2:z:0*
T0*'
_output_shapes
:���������w
SelectV2SelectV2LessEqual:z:0LeakyRelu:activations:0mul:z:0*
T0*'
_output_shapes
:���������P
mul_1MulinputsCast:y:0*
T0*'
_output_shapes
:���������[
truedivRealDiv	mul_1:z:0
Cast_1:y:0*
T0*'
_output_shapes
:���������I
NegNegtruediv:z:0*
T0*'
_output_shapes
:���������M
RoundRoundtruediv:z:0*
T0*'
_output_shapes
:���������R
addAddV2Neg:y:0	Round:y:0*
T0*'
_output_shapes
:���������W
StopGradientStopGradientadd:z:0*
T0*'
_output_shapes
:���������d
add_1AddV2truediv:z:0StopGradient:output:0*
T0*'
_output_shapes
:���������[
	truediv_1RealDiv	add_1:z:0Cast:y:0*
T0*'
_output_shapes
:���������P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
	truediv_2RealDivtruediv_2/x:output:0Cast:y:0*
T0*
_output_shapes
: L
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_3Subsub_3/x:output:0truediv_2:z:0*
T0*
_output_shapes
: l
clip_by_value/MinimumMinimumtruediv_1:z:0	sub_3:z:0*
T0*'
_output_shapes
:���������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������]
mul_2Mul
Cast_1:y:0clip_by_value:z:0*
T0*'
_output_shapes
:���������L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �:I
mul_3Mulmul_3/x:output:0Cast:y:0*
T0*
_output_shapes
: P
truediv_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
	truediv_3RealDivtruediv_3/x:output:0	mul_3:z:0*
T0*
_output_shapes
: L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �:K
mul_4Mul
Cast_1:y:0mul_4/y:output:0*
T0*
_output_shapes
: L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �:]
mul_5Multruediv:z:0mul_5/y:output:0*
T0*'
_output_shapes
:���������I
Neg_1Neg	mul_5:z:0*
T0*'
_output_shapes
:���������M
Round_1Round	mul_5:z:0*
T0*'
_output_shapes
:���������X
add_2AddV2	Neg_1:y:0Round_1:y:0*
T0*'
_output_shapes
:���������[
StopGradient_1StopGradient	add_2:z:0*
T0*'
_output_shapes
:���������d
add_3AddV2	mul_5:z:0StopGradient_1:output:0*
T0*'
_output_shapes
:���������X
mul_6Mul	add_3:z:0truediv_3:z:0*
T0*'
_output_shapes
:���������^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_value_1/MinimumMinimum	mul_6:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:���������V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:���������^
mul_7Mul	mul_4:z:0clip_by_value_1:z:0*
T0*'
_output_shapes
:���������V
add_4AddV2	mul_2:z:0	mul_7:z:0*
T0*'
_output_shapes
:���������Q
Neg_2NegSelectV2:output:0*
T0*'
_output_shapes
:���������V
add_5AddV2	Neg_2:y:0	add_4:z:0*
T0*'
_output_shapes
:���������L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
mul_8Mulmul_8/x:output:0	add_5:z:0*
T0*'
_output_shapes
:���������[
StopGradient_2StopGradient	mul_8:z:0*
T0*'
_output_shapes
:���������l
add_6AddV2SelectV2:output:0StopGradient_2:output:0*
T0*'
_output_shapes
:���������Q
IdentityIdentity	add_6:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�s
�
 __inference__traced_save_8207142
file_prefix6
$read_disablecopyonread_dense1_kernel:, 2
$read_1_disablecopyonread_dense1_bias: 0
"read_2_disablecopyonread_bn1_gamma: /
!read_3_disablecopyonread_bn1_beta: 6
(read_4_disablecopyonread_bn1_moving_mean: :
,read_5_disablecopyonread_bn1_moving_variance: 8
&read_6_disablecopyonread_dense2_kernel: 2
$read_7_disablecopyonread_dense2_bias:0
"read_8_disablecopyonread_bn2_gamma:/
!read_9_disablecopyonread_bn2_beta:7
)read_10_disablecopyonread_bn2_moving_mean:;
-read_11_disablecopyonread_bn2_moving_variance:9
'read_12_disablecopyonread_z_mean_kernel:3
%read_13_disablecopyonread_z_mean_bias:
savev2_const
identity_29��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: v
Read/DisableCopyOnReadDisableCopyOnRead$read_disablecopyonread_dense1_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp$read_disablecopyonread_dense1_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:, *
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:, a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:, x
Read_1/DisableCopyOnReadDisableCopyOnRead$read_1_disablecopyonread_dense1_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp$read_1_disablecopyonread_dense1_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_2/DisableCopyOnReadDisableCopyOnRead"read_2_disablecopyonread_bn1_gamma"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp"read_2_disablecopyonread_bn1_gamma^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
: u
Read_3/DisableCopyOnReadDisableCopyOnRead!read_3_disablecopyonread_bn1_beta"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp!read_3_disablecopyonread_bn1_beta^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
: |
Read_4/DisableCopyOnReadDisableCopyOnRead(read_4_disablecopyonread_bn1_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp(read_4_disablecopyonread_bn1_moving_mean^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_5/DisableCopyOnReadDisableCopyOnRead,read_5_disablecopyonread_bn1_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp,read_5_disablecopyonread_bn1_moving_variance^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
: z
Read_6/DisableCopyOnReadDisableCopyOnRead&read_6_disablecopyonread_dense2_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp&read_6_disablecopyonread_dense2_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0n
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

: x
Read_7/DisableCopyOnReadDisableCopyOnRead$read_7_disablecopyonread_dense2_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp$read_7_disablecopyonread_dense2_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_8/DisableCopyOnReadDisableCopyOnRead"read_8_disablecopyonread_bn2_gamma"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp"read_8_disablecopyonread_bn2_gamma^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:u
Read_9/DisableCopyOnReadDisableCopyOnRead!read_9_disablecopyonread_bn2_beta"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp!read_9_disablecopyonread_bn2_beta^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_10/DisableCopyOnReadDisableCopyOnRead)read_10_disablecopyonread_bn2_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp)read_10_disablecopyonread_bn2_moving_mean^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_11/DisableCopyOnReadDisableCopyOnRead-read_11_disablecopyonread_bn2_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp-read_11_disablecopyonread_bn2_moving_variance^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_12/DisableCopyOnReadDisableCopyOnRead'read_12_disablecopyonread_z_mean_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp'read_12_disablecopyonread_z_mean_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes

:z
Read_13/DisableCopyOnReadDisableCopyOnRead%read_13_disablecopyonread_z_mean_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp%read_13_disablecopyonread_z_mean_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_28Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_29IdentityIdentity_28:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_29Identity_29:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_user_specified_namedense1/kernel:+'
%
_user_specified_namedense1/bias:)%
#
_user_specified_name	BN1/gamma:($
"
_user_specified_name
BN1/beta:/+
)
_user_specified_nameBN1/moving_mean:3/
-
_user_specified_nameBN1/moving_variance:-)
'
_user_specified_namedense2/kernel:+'
%
_user_specified_namedense2/bias:)	%
#
_user_specified_name	BN2/gamma:(
$
"
_user_specified_name
BN2/beta:/+
)
_user_specified_nameBN2/moving_mean:3/
-
_user_specified_nameBN2/moving_variance:-)
'
_user_specified_namez_mean/kernel:+'
%
_user_specified_namez_mean/bias:=9

_output_shapes
: 

_user_specified_nameConst
�2
�
C__inference_dense2_layer_call_and_return_conditional_losses_8203789

inputs)
readvariableop_resource: '
readvariableop_3_resource:
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

: *
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   D[
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes

: N
truedivRealDivmul:z:0Cast:y:0*
T0*
_output_shapes

: @
NegNegtruediv:z:0*
T0*
_output_shapes

: D
RoundRoundtruediv:z:0*
T0*
_output_shapes

: I
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

: N
StopGradientStopGradientadd:z:0*
T0*
_output_shapes

: [
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

: \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ��Cv
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

: T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

: R
mul_1MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

: P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   D^
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*
_output_shapes

: L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*
_output_shapes

: h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

: *
dtype0O
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes

: M
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*
_output_shapes

: L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*
_output_shapes

: R
StopGradient_1StopGradient	mul_3:z:0*
T0*
_output_shapes

: h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

: *
dtype0j
add_3AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*
_output_shapes

: U
MatMulMatMulinputs	add_3:z:0*
T0*'
_output_shapes
:���������I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   D]
mul_4MulReadVariableOp_3:value:0mul_4/y:output:0*
T0*
_output_shapes
:P
	truediv_2RealDiv	mul_4:z:0
Cast_1:y:0*
T0*
_output_shapes
:@
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_4AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_4:z:0*
T0*
_output_shapes
:[
add_5AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ��Cv
clip_by_value_1/MinimumMinimum	add_5:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:R
mul_5Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   DZ
	truediv_3RealDiv	mul_5:z:0truediv_3/y:output:0*
T0*
_output_shapes
:L
mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_6Mulmul_6/x:output:0truediv_3:z:0*
T0*
_output_shapes
:f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0K
Neg_3NegReadVariableOp_4:value:0*
T0*
_output_shapes
:I
add_6AddV2	Neg_3:y:0	mul_6:z:0*
T0*
_output_shapes
:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_7Mulmul_7/x:output:0	add_6:z:0*
T0*
_output_shapes
:N
StopGradient_3StopGradient	mul_7:z:0*
T0*
_output_shapes
:f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0f
add_7AddV2ReadVariableOp_5:value:0StopGradient_3:output:0*
T0*
_output_shapes
:a
BiasAddBiasAddMatMul:product:0	add_7:z:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_5:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�	
�
%__inference_BN2_layer_call_fn_8206172

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_BN2_layer_call_and_return_conditional_losses_8204973o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:'#
!
_user_specified_name	8206162:'#
!
_user_specified_name	8206164:'#
!
_user_specified_name	8206166:'#
!
_user_specified_name	8206168
�
�
4__inference_simplified_encoder_layer_call_fn_8205023

inputs
unknown:, 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_simplified_encoder_layer_call_and_return_conditional_losses_8204371o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������,: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������,
 
_user_specified_nameinputs:'#
!
_user_specified_name	8204993:'#
!
_user_specified_name	8204995:'#
!
_user_specified_name	8204997:'#
!
_user_specified_name	8204999:'#
!
_user_specified_name	8205001:'#
!
_user_specified_name	8205003:'#
!
_user_specified_name	8205005:'#
!
_user_specified_name	8205007:'	#
!
_user_specified_name	8205009:'
#
!
_user_specified_name	8205011:'#
!
_user_specified_name	8205013:'#
!
_user_specified_name	8205015:'#
!
_user_specified_name	8205017:'#
!
_user_specified_name	8205019
��
�
@__inference_BN1_layer_call_and_return_conditional_losses_8204668

inputs%
readvariableop_resource: '
readvariableop_3_resource: '
readvariableop_6_resource: ,
relu_3_readvariableop_resource: 
identity��Abs_1/ReadVariableOp�Abs_3/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_10�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�ReadVariableOp_6�ReadVariableOp_7�ReadVariableOp_8�ReadVariableOp_9�Relu/ReadVariableOp�Relu_1/ReadVariableOp�Relu_3/ReadVariableOp�Relu_4/ReadVariableOpb
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0P
LessEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ei
	LessEqual	LessEqualReadVariableOp:value:0LessEqual/y:output:0*
T0*
_output_shapes
: g
Relu/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0N
ReluReluRelu/ReadVariableOp:value:0*
T0*
_output_shapes
: l
ones_like/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0i
ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?z
	ones_likeFill(ones_like/Shape/shape_as_tensor:output:0ones_like/Const:output:0*
T0*
_output_shapes
: J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ES
mulMulones_like:output:0mul/y:output:0*
T0*
_output_shapes
: e
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*
_output_shapes
: i
Relu_1/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0R
Relu_1ReluRelu_1/ReadVariableOp:value:0*
T0*
_output_shapes
: K
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3X
LessLessRelu_1:activations:0Less/y:output:0*
T0*
_output_shapes
: Q
SelectV2_1/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3r

SelectV2_1SelectV2Less:z:0SelectV2_1/t:output:0Relu_1:activations:0*
T0*
_output_shapes
: S
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Eo
GreaterEqualGreaterEqualSelectV2_1:output:0GreaterEqual/y:output:0*
T0*
_output_shapes
: k
!ones_like_1/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_1Fill*ones_like_1/Shape/shape_as_tensor:output:0ones_like_1/Const:output:0*
T0*
_output_shapes
: L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   EY
mul_1Mulones_like_1:output:0mul_1/y:output:0*
T0*
_output_shapes
: m

SelectV2_2SelectV2GreaterEqual:z:0	mul_1:z:0SelectV2_1:output:0*
T0*
_output_shapes
: M
Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3\
Less_1LessRelu_1:activations:0Less_1/y:output:0*
T0*
_output_shapes
: k
!ones_like_2/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_2Fill*ones_like_2/Shape/shape_as_tensor:output:0ones_like_2/Const:output:0*
T0*
_output_shapes
: L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �Y
mul_2Mulones_like_2:output:0mul_2/y:output:0*
T0*
_output_shapes
: D
LogLogSelectV2_2:output:0*
T0*
_output_shapes
: N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?T
truedivRealDivLog:y:0truediv/y:output:0*
T0*
_output_shapes
: <
NegNegtruediv:z:0*
T0*
_output_shapes
: @
RoundRoundtruediv:z:0*
T0*
_output_shapes
: E
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes
: J
StopGradientStopGradientadd:z:0*
T0*
_output_shapes
: W
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes
: \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Ar
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
: T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �r
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes
: L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_3Mulmul_3/x:output:0clip_by_value:z:0*
T0*
_output_shapes
: ]

SelectV2_3SelectV2
Less_1:z:0	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: d
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0K
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes
: >
Relu_2Relu	Neg_1:y:0*
T0*
_output_shapes
: L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
mul_4MulRelu_2:activations:0mul_4/y:output:0*
T0*
_output_shapes
: M
Less_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_2Less	mul_4:z:0Less_2/y:output:0*
T0*
_output_shapes
: Q
SelectV2_4/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3i

SelectV2_4SelectV2
Less_2:z:0SelectV2_4/t:output:0	mul_4:z:0*
T0*
_output_shapes
: U
GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Es
GreaterEqual_1GreaterEqualSelectV2_4:output:0GreaterEqual_1/y:output:0*
T0*
_output_shapes
: k
!ones_like_3/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_3Fill*ones_like_3/Shape/shape_as_tensor:output:0ones_like_3/Const:output:0*
T0*
_output_shapes
: L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *   EY
mul_5Mulones_like_3:output:0mul_5/y:output:0*
T0*
_output_shapes
: o

SelectV2_5SelectV2GreaterEqual_1:z:0	mul_5:z:0SelectV2_4:output:0*
T0*
_output_shapes
: M
Less_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_3Less	mul_4:z:0Less_3/y:output:0*
T0*
_output_shapes
: k
!ones_like_4/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_4Fill*ones_like_4/Shape/shape_as_tensor:output:0ones_like_4/Const:output:0*
T0*
_output_shapes
: L
mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �Y
mul_6Mulones_like_4:output:0mul_6/y:output:0*
T0*
_output_shapes
: F
Log_1LogSelectV2_5:output:0*
T0*
_output_shapes
: P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_1RealDiv	Log_1:y:0truediv_1/y:output:0*
T0*
_output_shapes
: @
Neg_2Negtruediv_1:z:0*
T0*
_output_shapes
: D
Round_1Roundtruediv_1:z:0*
T0*
_output_shapes
: K
add_2AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
: N
StopGradient_1StopGradient	add_2:z:0*
T0*
_output_shapes
: [
add_3AddV2truediv_1:z:0StopGradient_1:output:0*
T0*
_output_shapes
: ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Av
clip_by_value_1/MinimumMinimum	add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
: V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
: L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?X
mul_7Mulmul_7/x:output:0clip_by_value_1:z:0*
T0*
_output_shapes
: ]

SelectV2_6SelectV2
Less_3:z:0	mul_6:z:0	mul_7:z:0*
T0*
_output_shapes
: d
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0U
GreaterEqual_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    x
GreaterEqual_2GreaterEqualReadVariableOp_2:value:0GreaterEqual_2/y:output:0*
T0*
_output_shapes
: M
LogicalOr/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z\
	LogicalOr	LogicalOrGreaterEqual_2:z:0LogicalOr/y:output:0*
_output_shapes
: J
pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
powPowpow/x:output:0SelectV2_3:output:0*
T0*
_output_shapes
: L
pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @X
pow_1Powpow_1/x:output:0SelectV2_6:output:0*
T0*
_output_shapes
: <
Neg_3Neg	pow_1:z:0*
T0*
_output_shapes
: ^

SelectV2_7SelectV2LogicalOr:z:0pow:z:0	Neg_3:y:0*
T0*
_output_shapes
: D
Neg_4NegSelectV2:output:0*
T0*
_output_shapes
: S
add_4AddV2	Neg_4:y:0SelectV2_7:output:0*
T0*
_output_shapes
: L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_8Mulmul_8/x:output:0	add_4:z:0*
T0*
_output_shapes
: N
StopGradient_2StopGradient	mul_8:z:0*
T0*
_output_shapes
: _
add_5AddV2SelectV2:output:0StopGradient_2:output:0*
T0*
_output_shapes
: f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
: *
dtype0K
SignSignReadVariableOp_3:value:0*
T0*
_output_shapes
: 9
AbsAbsSign:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?H
subSubsub/x:output:0Abs:y:0*
T0*
_output_shapes
: F
add_6AddV2Sign:y:0sub:z:0*
T0*
_output_shapes
: j
Abs_1/ReadVariableOpReadVariableOpreadvariableop_3_resource*
_output_shapes
: *
dtype0O
Abs_1AbsAbs_1/ReadVariableOp:value:0*
T0*
_output_shapes
: M
Less_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_4Less	Abs_1:y:0Less_4/y:output:0*
T0*
_output_shapes
: Q
SelectV2_8/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3i

SelectV2_8SelectV2
Less_4:z:0SelectV2_8/t:output:0	Abs_1:y:0*
T0*
_output_shapes
: M
Less_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_5Less	Abs_1:y:0Less_5/y:output:0*
T0*
_output_shapes
: k
!ones_like_5/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_5Fill*ones_like_5/Shape/shape_as_tensor:output:0ones_like_5/Const:output:0*
T0*
_output_shapes
: L
mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �Y
mul_9Mulones_like_5:output:0mul_9/y:output:0*
T0*
_output_shapes
: F
Log_2LogSelectV2_8:output:0*
T0*
_output_shapes
: P
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_2RealDiv	Log_2:y:0truediv_2/y:output:0*
T0*
_output_shapes
: @
Neg_5Negtruediv_2:z:0*
T0*
_output_shapes
: D
Round_2Roundtruediv_2:z:0*
T0*
_output_shapes
: K
add_7AddV2	Neg_5:y:0Round_2:y:0*
T0*
_output_shapes
: N
StopGradient_3StopGradient	add_7:z:0*
T0*
_output_shapes
: [
add_8AddV2truediv_2:z:0StopGradient_3:output:0*
T0*
_output_shapes
: ^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@v
clip_by_value_2/MinimumMinimum	add_8:z:0"clip_by_value_2/Minimum/y:output:0*
T0*
_output_shapes
: V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*
_output_shapes
: M
mul_10/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
mul_10Mulmul_10/x:output:0clip_by_value_2:z:0*
T0*
_output_shapes
: ^

SelectV2_9SelectV2
Less_5:z:0	mul_9:z:0
mul_10:z:0*
T0*
_output_shapes
: L
pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @X
pow_2Powpow_2/x:output:0SelectV2_9:output:0*
T0*
_output_shapes
: H
mul_11Mul	add_6:z:0	pow_2:z:0*
T0*
_output_shapes
: f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
: *
dtype0K
Neg_6NegReadVariableOp_4:value:0*
T0*
_output_shapes
: J
add_9AddV2	Neg_6:y:0
mul_11:z:0*
T0*
_output_shapes
: M
mul_12/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?P
mul_12Mulmul_12/x:output:0	add_9:z:0*
T0*
_output_shapes
: O
StopGradient_4StopGradient
mul_12:z:0*
T0*
_output_shapes
: f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
: *
dtype0g
add_10AddV2ReadVariableOp_5:value:0StopGradient_4:output:0*
T0*
_output_shapes
: f
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource*
_output_shapes
: *
dtype0M
Sign_1SignReadVariableOp_6:value:0*
T0*
_output_shapes
: =
Abs_2Abs
Sign_1:y:0*
T0*
_output_shapes
: L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_1Subsub_1/x:output:0	Abs_2:y:0*
T0*
_output_shapes
: K
add_11AddV2
Sign_1:y:0	sub_1:z:0*
T0*
_output_shapes
: j
Abs_3/ReadVariableOpReadVariableOpreadvariableop_6_resource*
_output_shapes
: *
dtype0O
Abs_3AbsAbs_3/ReadVariableOp:value:0*
T0*
_output_shapes
: M
Less_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_6Less	Abs_3:y:0Less_6/y:output:0*
T0*
_output_shapes
: R
SelectV2_10/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3k
SelectV2_10SelectV2
Less_6:z:0SelectV2_10/t:output:0	Abs_3:y:0*
T0*
_output_shapes
: M
Less_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_7Less	Abs_3:y:0Less_7/y:output:0*
T0*
_output_shapes
: k
!ones_like_6/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_6Fill*ones_like_6/Shape/shape_as_tensor:output:0ones_like_6/Const:output:0*
T0*
_output_shapes
: M
mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �[
mul_13Mulones_like_6:output:0mul_13/y:output:0*
T0*
_output_shapes
: G
Log_3LogSelectV2_10:output:0*
T0*
_output_shapes
: P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_3RealDiv	Log_3:y:0truediv_3/y:output:0*
T0*
_output_shapes
: @
Neg_7Negtruediv_3:z:0*
T0*
_output_shapes
: D
Round_3Roundtruediv_3:z:0*
T0*
_output_shapes
: L
add_12AddV2	Neg_7:y:0Round_3:y:0*
T0*
_output_shapes
: O
StopGradient_5StopGradient
add_12:z:0*
T0*
_output_shapes
: \
add_13AddV2truediv_3:z:0StopGradient_5:output:0*
T0*
_output_shapes
: ^
clip_by_value_3/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@w
clip_by_value_3/MinimumMinimum
add_13:z:0"clip_by_value_3/Minimum/y:output:0*
T0*
_output_shapes
: V
clip_by_value_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0*
_output_shapes
: M
mul_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
mul_14Mulmul_14/x:output:0clip_by_value_3:z:0*
T0*
_output_shapes
: `
SelectV2_11SelectV2
Less_7:z:0
mul_13:z:0
mul_14:z:0*
T0*
_output_shapes
: L
pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_3Powpow_3/x:output:0SelectV2_11:output:0*
T0*
_output_shapes
: I
mul_15Mul
add_11:z:0	pow_3:z:0*
T0*
_output_shapes
: f
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource*
_output_shapes
: *
dtype0K
Neg_8NegReadVariableOp_7:value:0*
T0*
_output_shapes
: K
add_14AddV2	Neg_8:y:0
mul_15:z:0*
T0*
_output_shapes
: M
mul_16/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
mul_16Mulmul_16/x:output:0
add_14:z:0*
T0*
_output_shapes
: O
StopGradient_6StopGradient
mul_16:z:0*
T0*
_output_shapes
: f
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource*
_output_shapes
: *
dtype0g
add_15AddV2ReadVariableOp_8:value:0StopGradient_6:output:0*
T0*
_output_shapes
: p
Relu_3/ReadVariableOpReadVariableOprelu_3_readvariableop_resource*
_output_shapes
: *
dtype0R
Relu_3ReluRelu_3/ReadVariableOp:value:0*
T0*
_output_shapes
: p
Relu_4/ReadVariableOpReadVariableOprelu_3_readvariableop_resource*
_output_shapes
: *
dtype0R
Relu_4ReluRelu_4/ReadVariableOp:value:0*
T0*
_output_shapes
: M
Less_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3\
Less_8LessRelu_4:activations:0Less_8/y:output:0*
T0*
_output_shapes
: R
SelectV2_12/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3v
SelectV2_12SelectV2
Less_8:z:0SelectV2_12/t:output:0Relu_4:activations:0*
T0*
_output_shapes
: M
Less_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3\
Less_9LessRelu_4:activations:0Less_9/y:output:0*
T0*
_output_shapes
: k
!ones_like_7/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_7Fill*ones_like_7/Shape/shape_as_tensor:output:0ones_like_7/Const:output:0*
T0*
_output_shapes
: M
mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �[
mul_17Mulones_like_7:output:0mul_17/y:output:0*
T0*
_output_shapes
: G
SqrtSqrtSelectV2_12:output:0*
T0*
_output_shapes
: ;
Log_4LogSqrt:y:0*
T0*
_output_shapes
: P
truediv_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_4RealDiv	Log_4:y:0truediv_4/y:output:0*
T0*
_output_shapes
: @
Neg_9Negtruediv_4:z:0*
T0*
_output_shapes
: D
Round_4Roundtruediv_4:z:0*
T0*
_output_shapes
: L
add_16AddV2	Neg_9:y:0Round_4:y:0*
T0*
_output_shapes
: O
StopGradient_7StopGradient
add_16:z:0*
T0*
_output_shapes
: \
add_17AddV2truediv_4:z:0StopGradient_7:output:0*
T0*
_output_shapes
: ^
clip_by_value_4/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Aw
clip_by_value_4/MinimumMinimum
add_17:z:0"clip_by_value_4/Minimum/y:output:0*
T0*
_output_shapes
: V
clip_by_value_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_4Maximumclip_by_value_4/Minimum:z:0clip_by_value_4/y:output:0*
T0*
_output_shapes
: M
mul_18/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Z
mul_18Mulmul_18/x:output:0clip_by_value_4:z:0*
T0*
_output_shapes
: `
SelectV2_13SelectV2
Less_9:z:0
mul_17:z:0
mul_18:z:0*
T0*
_output_shapes
: k
ReadVariableOp_9ReadVariableOprelu_3_readvariableop_resource*
_output_shapes
: *
dtype0L
Neg_10NegReadVariableOp_9:value:0*
T0*
_output_shapes
: ?
Relu_5Relu
Neg_10:y:0*
T0*
_output_shapes
: M
mul_19/yConst*
_output_shapes
: *
dtype0*
valueB
 *    [
mul_19MulRelu_5:activations:0mul_19/y:output:0*
T0*
_output_shapes
: N
	Less_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3T
Less_10Less
mul_19:z:0Less_10/y:output:0*
T0*
_output_shapes
: R
SelectV2_14/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3m
SelectV2_14SelectV2Less_10:z:0SelectV2_14/t:output:0
mul_19:z:0*
T0*
_output_shapes
: N
	Less_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3T
Less_11Less
mul_19:z:0Less_11/y:output:0*
T0*
_output_shapes
: k
!ones_like_8/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_8Fill*ones_like_8/Shape/shape_as_tensor:output:0ones_like_8/Const:output:0*
T0*
_output_shapes
: M
mul_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �[
mul_20Mulones_like_8:output:0mul_20/y:output:0*
T0*
_output_shapes
: I
Sqrt_1SqrtSelectV2_14:output:0*
T0*
_output_shapes
: =
Log_5Log
Sqrt_1:y:0*
T0*
_output_shapes
: P
truediv_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_5RealDiv	Log_5:y:0truediv_5/y:output:0*
T0*
_output_shapes
: A
Neg_11Negtruediv_5:z:0*
T0*
_output_shapes
: D
Round_5Roundtruediv_5:z:0*
T0*
_output_shapes
: M
add_18AddV2
Neg_11:y:0Round_5:y:0*
T0*
_output_shapes
: O
StopGradient_8StopGradient
add_18:z:0*
T0*
_output_shapes
: \
add_19AddV2truediv_5:z:0StopGradient_8:output:0*
T0*
_output_shapes
: ^
clip_by_value_5/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Aw
clip_by_value_5/MinimumMinimum
add_19:z:0"clip_by_value_5/Minimum/y:output:0*
T0*
_output_shapes
: V
clip_by_value_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_5Maximumclip_by_value_5/Minimum:z:0clip_by_value_5/y:output:0*
T0*
_output_shapes
: M
mul_21/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Z
mul_21Mulmul_21/x:output:0clip_by_value_5:z:0*
T0*
_output_shapes
: a
SelectV2_15SelectV2Less_11:z:0
mul_20:z:0
mul_21:z:0*
T0*
_output_shapes
: l
ReadVariableOp_10ReadVariableOprelu_3_readvariableop_resource*
_output_shapes
: *
dtype0U
GreaterEqual_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *    y
GreaterEqual_3GreaterEqualReadVariableOp_10:value:0GreaterEqual_3/y:output:0*
T0*
_output_shapes
: O
LogicalOr_1/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z`
LogicalOr_1	LogicalOrGreaterEqual_3:z:0LogicalOr_1/y:output:0*
_output_shapes
: L
pow_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_4Powpow_4/x:output:0SelectV2_13:output:0*
T0*
_output_shapes
: L
pow_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_5Powpow_5/x:output:0SelectV2_15:output:0*
T0*
_output_shapes
: =
Neg_12Neg	pow_5:z:0*
T0*
_output_shapes
: d
SelectV2_16SelectV2LogicalOr_1:z:0	pow_4:z:0
Neg_12:y:0*
T0*
_output_shapes
: H
Neg_13NegRelu_3:activations:0*
T0*
_output_shapes
: V
add_20AddV2
Neg_13:y:0SelectV2_16:output:0*
T0*
_output_shapes
: M
mul_22/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
mul_22Mulmul_22/x:output:0
add_20:z:0*
T0*
_output_shapes
: O
StopGradient_9StopGradient
mul_22:z:0*
T0*
_output_shapes
: c
add_21AddV2Relu_3:activations:0StopGradient_9:output:0*
T0*
_output_shapes
: T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:a
batchnorm/addAddV2
add_21:z:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: Y
batchnorm/mulMulbatchnorm/Rsqrt:y:0	add_5:z:0*
T0*
_output_shapes
: c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:��������� Z
batchnorm/mul_2Mul
add_15:z:0batchnorm/mul:z:0*
T0*
_output_shapes
: Z
batchnorm/subSub
add_10:z:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^Abs_1/ReadVariableOp^Abs_3/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_10^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8^ReadVariableOp_9^Relu/ReadVariableOp^Relu_1/ReadVariableOp^Relu_3/ReadVariableOp^Relu_4/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 2,
Abs_1/ReadVariableOpAbs_1/ReadVariableOp2,
Abs_3/ReadVariableOpAbs_3/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12&
ReadVariableOp_10ReadVariableOp_102$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82$
ReadVariableOp_9ReadVariableOp_92*
Relu/ReadVariableOpRelu/ReadVariableOp2.
Relu_1/ReadVariableOpRelu_1/ReadVariableOp2.
Relu_3/ReadVariableOpRelu_3/ReadVariableOp2.
Relu_4/ReadVariableOpRelu_4/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�2
�
C__inference_dense1_layer_call_and_return_conditional_losses_8205256

inputs)
readvariableop_resource:, '
readvariableop_3_resource: 
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:, *
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   D[
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes

:, N
truedivRealDivmul:z:0Cast:y:0*
T0*
_output_shapes

:, @
NegNegtruediv:z:0*
T0*
_output_shapes

:, D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:, I
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:, N
StopGradientStopGradientadd:z:0*
T0*
_output_shapes

:, [
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:, \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ��Cv
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:, T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:, R
mul_1MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

:, P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   D^
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*
_output_shapes

:, L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*
_output_shapes

:, h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:, *
dtype0O
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes

:, M
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*
_output_shapes

:, L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*
_output_shapes

:, R
StopGradient_1StopGradient	mul_3:z:0*
T0*
_output_shapes

:, h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:, *
dtype0j
add_3AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*
_output_shapes

:, U
MatMulMatMulinputs	add_3:z:0*
T0*'
_output_shapes
:��������� I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
: *
dtype0L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   D]
mul_4MulReadVariableOp_3:value:0mul_4/y:output:0*
T0*
_output_shapes
: P
	truediv_2RealDiv	mul_4:z:0
Cast_1:y:0*
T0*
_output_shapes
: @
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
: D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
: K
add_4AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
: N
StopGradient_2StopGradient	add_4:z:0*
T0*
_output_shapes
: [
add_5AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
: ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ��Cv
clip_by_value_1/MinimumMinimum	add_5:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
: V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
: R
mul_5Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
: P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   DZ
	truediv_3RealDiv	mul_5:z:0truediv_3/y:output:0*
T0*
_output_shapes
: L
mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_6Mulmul_6/x:output:0truediv_3:z:0*
T0*
_output_shapes
: f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
: *
dtype0K
Neg_3NegReadVariableOp_4:value:0*
T0*
_output_shapes
: I
add_6AddV2	Neg_3:y:0	mul_6:z:0*
T0*
_output_shapes
: L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_7Mulmul_7/x:output:0	add_6:z:0*
T0*
_output_shapes
: N
StopGradient_3StopGradient	mul_7:z:0*
T0*
_output_shapes
: f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
: *
dtype0f
add_7AddV2ReadVariableOp_5:value:0StopGradient_3:output:0*
T0*
_output_shapes
: a
BiasAddBiasAddMatMul:product:0	add_7:z:0*
T0*'
_output_shapes
:��������� _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������,: : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_5:O K
'
_output_shapes
:���������,
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
��
�
@__inference_BN2_layer_call_and_return_conditional_losses_8206595

inputs%
readvariableop_resource:'
readvariableop_3_resource:'
readvariableop_6_resource:,
relu_3_readvariableop_resource:
identity��Abs_1/ReadVariableOp�Abs_3/ReadVariableOp�AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_10�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�ReadVariableOp_6�ReadVariableOp_7�ReadVariableOp_8�ReadVariableOp_9�Relu/ReadVariableOp�Relu_1/ReadVariableOp�Relu_3/ReadVariableOp�Relu_4/ReadVariableOpb
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0P
LessEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ei
	LessEqual	LessEqualReadVariableOp:value:0LessEqual/y:output:0*
T0*
_output_shapes
:g
Relu/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0N
ReluReluRelu/ReadVariableOp:value:0*
T0*
_output_shapes
:l
ones_like/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0i
ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?z
	ones_likeFill(ones_like/Shape/shape_as_tensor:output:0ones_like/Const:output:0*
T0*
_output_shapes
:J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ES
mulMulones_like:output:0mul/y:output:0*
T0*
_output_shapes
:e
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*
_output_shapes
:i
Relu_1/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0R
Relu_1ReluRelu_1/ReadVariableOp:value:0*
T0*
_output_shapes
:K
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3X
LessLessRelu_1:activations:0Less/y:output:0*
T0*
_output_shapes
:Q
SelectV2_1/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3r

SelectV2_1SelectV2Less:z:0SelectV2_1/t:output:0Relu_1:activations:0*
T0*
_output_shapes
:S
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Eo
GreaterEqualGreaterEqualSelectV2_1:output:0GreaterEqual/y:output:0*
T0*
_output_shapes
:k
!ones_like_1/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_1Fill*ones_like_1/Shape/shape_as_tensor:output:0ones_like_1/Const:output:0*
T0*
_output_shapes
:L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   EY
mul_1Mulones_like_1:output:0mul_1/y:output:0*
T0*
_output_shapes
:m

SelectV2_2SelectV2GreaterEqual:z:0	mul_1:z:0SelectV2_1:output:0*
T0*
_output_shapes
:M
Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3\
Less_1LessRelu_1:activations:0Less_1/y:output:0*
T0*
_output_shapes
:k
!ones_like_2/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_2Fill*ones_like_2/Shape/shape_as_tensor:output:0ones_like_2/Const:output:0*
T0*
_output_shapes
:L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �Y
mul_2Mulones_like_2:output:0mul_2/y:output:0*
T0*
_output_shapes
:D
LogLogSelectV2_2:output:0*
T0*
_output_shapes
:N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?T
truedivRealDivLog:y:0truediv/y:output:0*
T0*
_output_shapes
:<
NegNegtruediv:z:0*
T0*
_output_shapes
:@
RoundRoundtruediv:z:0*
T0*
_output_shapes
:E
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes
:J
StopGradientStopGradientadd:z:0*
T0*
_output_shapes
:W
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes
:\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Ar
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �r
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes
:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_3Mulmul_3/x:output:0clip_by_value:z:0*
T0*
_output_shapes
:]

SelectV2_3SelectV2
Less_1:z:0	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:d
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0K
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes
:>
Relu_2Relu	Neg_1:y:0*
T0*
_output_shapes
:L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
mul_4MulRelu_2:activations:0mul_4/y:output:0*
T0*
_output_shapes
:M
Less_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_2Less	mul_4:z:0Less_2/y:output:0*
T0*
_output_shapes
:Q
SelectV2_4/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3i

SelectV2_4SelectV2
Less_2:z:0SelectV2_4/t:output:0	mul_4:z:0*
T0*
_output_shapes
:U
GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Es
GreaterEqual_1GreaterEqualSelectV2_4:output:0GreaterEqual_1/y:output:0*
T0*
_output_shapes
:k
!ones_like_3/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_3Fill*ones_like_3/Shape/shape_as_tensor:output:0ones_like_3/Const:output:0*
T0*
_output_shapes
:L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *   EY
mul_5Mulones_like_3:output:0mul_5/y:output:0*
T0*
_output_shapes
:o

SelectV2_5SelectV2GreaterEqual_1:z:0	mul_5:z:0SelectV2_4:output:0*
T0*
_output_shapes
:M
Less_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_3Less	mul_4:z:0Less_3/y:output:0*
T0*
_output_shapes
:k
!ones_like_4/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_4Fill*ones_like_4/Shape/shape_as_tensor:output:0ones_like_4/Const:output:0*
T0*
_output_shapes
:L
mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �Y
mul_6Mulones_like_4:output:0mul_6/y:output:0*
T0*
_output_shapes
:F
Log_1LogSelectV2_5:output:0*
T0*
_output_shapes
:P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_1RealDiv	Log_1:y:0truediv_1/y:output:0*
T0*
_output_shapes
:@
Neg_2Negtruediv_1:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_1:z:0*
T0*
_output_shapes
:K
add_2AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_1StopGradient	add_2:z:0*
T0*
_output_shapes
:[
add_3AddV2truediv_1:z:0StopGradient_1:output:0*
T0*
_output_shapes
:^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Av
clip_by_value_1/MinimumMinimum	add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?X
mul_7Mulmul_7/x:output:0clip_by_value_1:z:0*
T0*
_output_shapes
:]

SelectV2_6SelectV2
Less_3:z:0	mul_6:z:0	mul_7:z:0*
T0*
_output_shapes
:d
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0U
GreaterEqual_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    x
GreaterEqual_2GreaterEqualReadVariableOp_2:value:0GreaterEqual_2/y:output:0*
T0*
_output_shapes
:M
LogicalOr/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z\
	LogicalOr	LogicalOrGreaterEqual_2:z:0LogicalOr/y:output:0*
_output_shapes
:J
pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
powPowpow/x:output:0SelectV2_3:output:0*
T0*
_output_shapes
:L
pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @X
pow_1Powpow_1/x:output:0SelectV2_6:output:0*
T0*
_output_shapes
:<
Neg_3Neg	pow_1:z:0*
T0*
_output_shapes
:^

SelectV2_7SelectV2LogicalOr:z:0pow:z:0	Neg_3:y:0*
T0*
_output_shapes
:D
Neg_4NegSelectV2:output:0*
T0*
_output_shapes
:S
add_4AddV2	Neg_4:y:0SelectV2_7:output:0*
T0*
_output_shapes
:L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_8Mulmul_8/x:output:0	add_4:z:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	mul_8:z:0*
T0*
_output_shapes
:_
add_5AddV2SelectV2:output:0StopGradient_2:output:0*
T0*
_output_shapes
:f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0K
SignSignReadVariableOp_3:value:0*
T0*
_output_shapes
:9
AbsAbsSign:y:0*
T0*
_output_shapes
:J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?H
subSubsub/x:output:0Abs:y:0*
T0*
_output_shapes
:F
add_6AddV2Sign:y:0sub:z:0*
T0*
_output_shapes
:j
Abs_1/ReadVariableOpReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0O
Abs_1AbsAbs_1/ReadVariableOp:value:0*
T0*
_output_shapes
:M
Less_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_4Less	Abs_1:y:0Less_4/y:output:0*
T0*
_output_shapes
:Q
SelectV2_8/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3i

SelectV2_8SelectV2
Less_4:z:0SelectV2_8/t:output:0	Abs_1:y:0*
T0*
_output_shapes
:M
Less_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_5Less	Abs_1:y:0Less_5/y:output:0*
T0*
_output_shapes
:k
!ones_like_5/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_5Fill*ones_like_5/Shape/shape_as_tensor:output:0ones_like_5/Const:output:0*
T0*
_output_shapes
:L
mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �Y
mul_9Mulones_like_5:output:0mul_9/y:output:0*
T0*
_output_shapes
:F
Log_2LogSelectV2_8:output:0*
T0*
_output_shapes
:P
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_2RealDiv	Log_2:y:0truediv_2/y:output:0*
T0*
_output_shapes
:@
Neg_5Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_2Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_7AddV2	Neg_5:y:0Round_2:y:0*
T0*
_output_shapes
:N
StopGradient_3StopGradient	add_7:z:0*
T0*
_output_shapes
:[
add_8AddV2truediv_2:z:0StopGradient_3:output:0*
T0*
_output_shapes
:^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@v
clip_by_value_2/MinimumMinimum	add_8:z:0"clip_by_value_2/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*
_output_shapes
:M
mul_10/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
mul_10Mulmul_10/x:output:0clip_by_value_2:z:0*
T0*
_output_shapes
:^

SelectV2_9SelectV2
Less_5:z:0	mul_9:z:0
mul_10:z:0*
T0*
_output_shapes
:L
pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @X
pow_2Powpow_2/x:output:0SelectV2_9:output:0*
T0*
_output_shapes
:H
mul_11Mul	add_6:z:0	pow_2:z:0*
T0*
_output_shapes
:f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0K
Neg_6NegReadVariableOp_4:value:0*
T0*
_output_shapes
:J
add_9AddV2	Neg_6:y:0
mul_11:z:0*
T0*
_output_shapes
:M
mul_12/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?P
mul_12Mulmul_12/x:output:0	add_9:z:0*
T0*
_output_shapes
:O
StopGradient_4StopGradient
mul_12:z:0*
T0*
_output_shapes
:f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0g
add_10AddV2ReadVariableOp_5:value:0StopGradient_4:output:0*
T0*
_output_shapes
:f
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource*
_output_shapes
:*
dtype0M
Sign_1SignReadVariableOp_6:value:0*
T0*
_output_shapes
:=
Abs_2Abs
Sign_1:y:0*
T0*
_output_shapes
:L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_1Subsub_1/x:output:0	Abs_2:y:0*
T0*
_output_shapes
:K
add_11AddV2
Sign_1:y:0	sub_1:z:0*
T0*
_output_shapes
:j
Abs_3/ReadVariableOpReadVariableOpreadvariableop_6_resource*
_output_shapes
:*
dtype0O
Abs_3AbsAbs_3/ReadVariableOp:value:0*
T0*
_output_shapes
:M
Less_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_6Less	Abs_3:y:0Less_6/y:output:0*
T0*
_output_shapes
:R
SelectV2_10/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3k
SelectV2_10SelectV2
Less_6:z:0SelectV2_10/t:output:0	Abs_3:y:0*
T0*
_output_shapes
:M
Less_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_7Less	Abs_3:y:0Less_7/y:output:0*
T0*
_output_shapes
:k
!ones_like_6/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_6Fill*ones_like_6/Shape/shape_as_tensor:output:0ones_like_6/Const:output:0*
T0*
_output_shapes
:M
mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �[
mul_13Mulones_like_6:output:0mul_13/y:output:0*
T0*
_output_shapes
:G
Log_3LogSelectV2_10:output:0*
T0*
_output_shapes
:P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_3RealDiv	Log_3:y:0truediv_3/y:output:0*
T0*
_output_shapes
:@
Neg_7Negtruediv_3:z:0*
T0*
_output_shapes
:D
Round_3Roundtruediv_3:z:0*
T0*
_output_shapes
:L
add_12AddV2	Neg_7:y:0Round_3:y:0*
T0*
_output_shapes
:O
StopGradient_5StopGradient
add_12:z:0*
T0*
_output_shapes
:\
add_13AddV2truediv_3:z:0StopGradient_5:output:0*
T0*
_output_shapes
:^
clip_by_value_3/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@w
clip_by_value_3/MinimumMinimum
add_13:z:0"clip_by_value_3/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0*
_output_shapes
:M
mul_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
mul_14Mulmul_14/x:output:0clip_by_value_3:z:0*
T0*
_output_shapes
:`
SelectV2_11SelectV2
Less_7:z:0
mul_13:z:0
mul_14:z:0*
T0*
_output_shapes
:L
pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_3Powpow_3/x:output:0SelectV2_11:output:0*
T0*
_output_shapes
:I
mul_15Mul
add_11:z:0	pow_3:z:0*
T0*
_output_shapes
:f
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource*
_output_shapes
:*
dtype0K
Neg_8NegReadVariableOp_7:value:0*
T0*
_output_shapes
:K
add_14AddV2	Neg_8:y:0
mul_15:z:0*
T0*
_output_shapes
:M
mul_16/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
mul_16Mulmul_16/x:output:0
add_14:z:0*
T0*
_output_shapes
:O
StopGradient_6StopGradient
mul_16:z:0*
T0*
_output_shapes
:f
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource*
_output_shapes
:*
dtype0g
add_15AddV2ReadVariableOp_8:value:0StopGradient_6:output:0*
T0*
_output_shapes
:p
Relu_3/ReadVariableOpReadVariableOprelu_3_readvariableop_resource*
_output_shapes
:*
dtype0R
Relu_3ReluRelu_3/ReadVariableOp:value:0*
T0*
_output_shapes
:p
Relu_4/ReadVariableOpReadVariableOprelu_3_readvariableop_resource*
_output_shapes
:*
dtype0R
Relu_4ReluRelu_4/ReadVariableOp:value:0*
T0*
_output_shapes
:M
Less_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3\
Less_8LessRelu_4:activations:0Less_8/y:output:0*
T0*
_output_shapes
:R
SelectV2_12/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3v
SelectV2_12SelectV2
Less_8:z:0SelectV2_12/t:output:0Relu_4:activations:0*
T0*
_output_shapes
:M
Less_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3\
Less_9LessRelu_4:activations:0Less_9/y:output:0*
T0*
_output_shapes
:k
!ones_like_7/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_7Fill*ones_like_7/Shape/shape_as_tensor:output:0ones_like_7/Const:output:0*
T0*
_output_shapes
:M
mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �[
mul_17Mulones_like_7:output:0mul_17/y:output:0*
T0*
_output_shapes
:G
SqrtSqrtSelectV2_12:output:0*
T0*
_output_shapes
:;
Log_4LogSqrt:y:0*
T0*
_output_shapes
:P
truediv_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_4RealDiv	Log_4:y:0truediv_4/y:output:0*
T0*
_output_shapes
:@
Neg_9Negtruediv_4:z:0*
T0*
_output_shapes
:D
Round_4Roundtruediv_4:z:0*
T0*
_output_shapes
:L
add_16AddV2	Neg_9:y:0Round_4:y:0*
T0*
_output_shapes
:O
StopGradient_7StopGradient
add_16:z:0*
T0*
_output_shapes
:\
add_17AddV2truediv_4:z:0StopGradient_7:output:0*
T0*
_output_shapes
:^
clip_by_value_4/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Aw
clip_by_value_4/MinimumMinimum
add_17:z:0"clip_by_value_4/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_4Maximumclip_by_value_4/Minimum:z:0clip_by_value_4/y:output:0*
T0*
_output_shapes
:M
mul_18/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Z
mul_18Mulmul_18/x:output:0clip_by_value_4:z:0*
T0*
_output_shapes
:`
SelectV2_13SelectV2
Less_9:z:0
mul_17:z:0
mul_18:z:0*
T0*
_output_shapes
:k
ReadVariableOp_9ReadVariableOprelu_3_readvariableop_resource*
_output_shapes
:*
dtype0L
Neg_10NegReadVariableOp_9:value:0*
T0*
_output_shapes
:?
Relu_5Relu
Neg_10:y:0*
T0*
_output_shapes
:M
mul_19/yConst*
_output_shapes
: *
dtype0*
valueB
 *    [
mul_19MulRelu_5:activations:0mul_19/y:output:0*
T0*
_output_shapes
:N
	Less_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3T
Less_10Less
mul_19:z:0Less_10/y:output:0*
T0*
_output_shapes
:R
SelectV2_14/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3m
SelectV2_14SelectV2Less_10:z:0SelectV2_14/t:output:0
mul_19:z:0*
T0*
_output_shapes
:N
	Less_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3T
Less_11Less
mul_19:z:0Less_11/y:output:0*
T0*
_output_shapes
:k
!ones_like_8/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_8Fill*ones_like_8/Shape/shape_as_tensor:output:0ones_like_8/Const:output:0*
T0*
_output_shapes
:M
mul_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �[
mul_20Mulones_like_8:output:0mul_20/y:output:0*
T0*
_output_shapes
:I
Sqrt_1SqrtSelectV2_14:output:0*
T0*
_output_shapes
:=
Log_5Log
Sqrt_1:y:0*
T0*
_output_shapes
:P
truediv_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_5RealDiv	Log_5:y:0truediv_5/y:output:0*
T0*
_output_shapes
:A
Neg_11Negtruediv_5:z:0*
T0*
_output_shapes
:D
Round_5Roundtruediv_5:z:0*
T0*
_output_shapes
:M
add_18AddV2
Neg_11:y:0Round_5:y:0*
T0*
_output_shapes
:O
StopGradient_8StopGradient
add_18:z:0*
T0*
_output_shapes
:\
add_19AddV2truediv_5:z:0StopGradient_8:output:0*
T0*
_output_shapes
:^
clip_by_value_5/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Aw
clip_by_value_5/MinimumMinimum
add_19:z:0"clip_by_value_5/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_5Maximumclip_by_value_5/Minimum:z:0clip_by_value_5/y:output:0*
T0*
_output_shapes
:M
mul_21/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Z
mul_21Mulmul_21/x:output:0clip_by_value_5:z:0*
T0*
_output_shapes
:a
SelectV2_15SelectV2Less_11:z:0
mul_20:z:0
mul_21:z:0*
T0*
_output_shapes
:l
ReadVariableOp_10ReadVariableOprelu_3_readvariableop_resource*
_output_shapes
:*
dtype0U
GreaterEqual_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *    y
GreaterEqual_3GreaterEqualReadVariableOp_10:value:0GreaterEqual_3/y:output:0*
T0*
_output_shapes
:O
LogicalOr_1/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z`
LogicalOr_1	LogicalOrGreaterEqual_3:z:0LogicalOr_1/y:output:0*
_output_shapes
:L
pow_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_4Powpow_4/x:output:0SelectV2_13:output:0*
T0*
_output_shapes
:L
pow_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_5Powpow_5/x:output:0SelectV2_15:output:0*
T0*
_output_shapes
:=
Neg_12Neg	pow_5:z:0*
T0*
_output_shapes
:d
SelectV2_16SelectV2LogicalOr_1:z:0	pow_4:z:0
Neg_12:y:0*
T0*
_output_shapes
:H
Neg_13NegRelu_3:activations:0*
T0*
_output_shapes
:V
add_20AddV2
Neg_13:y:0SelectV2_16:output:0*
T0*
_output_shapes
:M
mul_22/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
mul_22Mulmul_22/x:output:0
add_20:z:0*
T0*
_output_shapes
:O
StopGradient_9StopGradient
mul_22:z:0*
T0*
_output_shapes
:c
add_21AddV2Relu_3:activations:0StopGradient_9:output:0*
T0*
_output_shapes
:h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 M
Sign_2Signmoments/Squeeze:output:0*
T0*
_output_shapes
:=
Abs_4Abs
Sign_2:y:0*
T0*
_output_shapes
:L
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_2Subsub_2/x:output:0	Abs_4:y:0*
T0*
_output_shapes
:K
add_22AddV2
Sign_2:y:0	sub_2:z:0*
T0*
_output_shapes
:K
Abs_5Absmoments/Squeeze:output:0*
T0*
_output_shapes
:N
	Less_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3S
Less_12Less	Abs_5:y:0Less_12/y:output:0*
T0*
_output_shapes
:R
SelectV2_17/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3l
SelectV2_17SelectV2Less_12:z:0SelectV2_17/t:output:0	Abs_5:y:0*
T0*
_output_shapes
:N
	Less_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3S
Less_13Less	Abs_5:y:0Less_13/y:output:0*
T0*
_output_shapes
:k
!ones_like_9/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_9/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_9Fill*ones_like_9/Shape/shape_as_tensor:output:0ones_like_9/Const:output:0*
T0*
_output_shapes
:M
mul_23/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �[
mul_23Mulones_like_9:output:0mul_23/y:output:0*
T0*
_output_shapes
:G
Log_6LogSelectV2_17:output:0*
T0*
_output_shapes
:P
truediv_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_6RealDiv	Log_6:y:0truediv_6/y:output:0*
T0*
_output_shapes
:A
Neg_14Negtruediv_6:z:0*
T0*
_output_shapes
:D
Round_6Roundtruediv_6:z:0*
T0*
_output_shapes
:M
add_23AddV2
Neg_14:y:0Round_6:y:0*
T0*
_output_shapes
:P
StopGradient_10StopGradient
add_23:z:0*
T0*
_output_shapes
:]
add_24AddV2truediv_6:z:0StopGradient_10:output:0*
T0*
_output_shapes
:^
clip_by_value_6/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@w
clip_by_value_6/MinimumMinimum
add_24:z:0"clip_by_value_6/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_6Maximumclip_by_value_6/Minimum:z:0clip_by_value_6/y:output:0*
T0*
_output_shapes
:M
mul_24/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
mul_24Mulmul_24/x:output:0clip_by_value_6:z:0*
T0*
_output_shapes
:a
SelectV2_18SelectV2Less_13:z:0
mul_23:z:0
mul_24:z:0*
T0*
_output_shapes
:L
pow_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_6Powpow_6/x:output:0SelectV2_18:output:0*
T0*
_output_shapes
:I
mul_25Mul
add_22:z:0	pow_6:z:0*
T0*
_output_shapes
:L
Neg_15Negmoments/Squeeze:output:0*
T0*
_output_shapes
:L
add_25AddV2
Neg_15:y:0
mul_25:z:0*
T0*
_output_shapes
:M
mul_26/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
mul_26Mulmul_26/x:output:0
add_25:z:0*
T0*
_output_shapes
:P
StopGradient_11StopGradient
mul_26:z:0*
T0*
_output_shapes
:h
add_26AddV2moments/Squeeze:output:0StopGradient_11:output:0*
T0*
_output_shapes
:O
Relu_6Relumoments/Squeeze_1:output:0*
T0*
_output_shapes
:O
Relu_7Relumoments/Squeeze_1:output:0*
T0*
_output_shapes
:N
	Less_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3^
Less_14LessRelu_7:activations:0Less_14/y:output:0*
T0*
_output_shapes
:R
SelectV2_19/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3w
SelectV2_19SelectV2Less_14:z:0SelectV2_19/t:output:0Relu_7:activations:0*
T0*
_output_shapes
:N
	Less_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3^
Less_15LessRelu_7:activations:0Less_15/y:output:0*
T0*
_output_shapes
:l
"ones_like_10/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:W
ones_like_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_10Fill+ones_like_10/Shape/shape_as_tensor:output:0ones_like_10/Const:output:0*
T0*
_output_shapes
:M
mul_27/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �\
mul_27Mulones_like_10:output:0mul_27/y:output:0*
T0*
_output_shapes
:I
Sqrt_2SqrtSelectV2_19:output:0*
T0*
_output_shapes
:=
Log_7Log
Sqrt_2:y:0*
T0*
_output_shapes
:P
truediv_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_7RealDiv	Log_7:y:0truediv_7/y:output:0*
T0*
_output_shapes
:A
Neg_16Negtruediv_7:z:0*
T0*
_output_shapes
:D
Round_7Roundtruediv_7:z:0*
T0*
_output_shapes
:M
add_27AddV2
Neg_16:y:0Round_7:y:0*
T0*
_output_shapes
:P
StopGradient_12StopGradient
add_27:z:0*
T0*
_output_shapes
:]
add_28AddV2truediv_7:z:0StopGradient_12:output:0*
T0*
_output_shapes
:^
clip_by_value_7/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Aw
clip_by_value_7/MinimumMinimum
add_28:z:0"clip_by_value_7/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_7Maximumclip_by_value_7/Minimum:z:0clip_by_value_7/y:output:0*
T0*
_output_shapes
:M
mul_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Z
mul_28Mulmul_28/x:output:0clip_by_value_7:z:0*
T0*
_output_shapes
:a
SelectV2_20SelectV2Less_15:z:0
mul_27:z:0
mul_28:z:0*
T0*
_output_shapes
:N
Neg_17Negmoments/Squeeze_1:output:0*
T0*
_output_shapes
:?
Relu_8Relu
Neg_17:y:0*
T0*
_output_shapes
:M
mul_29/yConst*
_output_shapes
: *
dtype0*
valueB
 *    [
mul_29MulRelu_8:activations:0mul_29/y:output:0*
T0*
_output_shapes
:N
	Less_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3T
Less_16Less
mul_29:z:0Less_16/y:output:0*
T0*
_output_shapes
:R
SelectV2_21/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3m
SelectV2_21SelectV2Less_16:z:0SelectV2_21/t:output:0
mul_29:z:0*
T0*
_output_shapes
:N
	Less_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3T
Less_17Less
mul_29:z:0Less_17/y:output:0*
T0*
_output_shapes
:l
"ones_like_11/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:W
ones_like_11/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_11Fill+ones_like_11/Shape/shape_as_tensor:output:0ones_like_11/Const:output:0*
T0*
_output_shapes
:M
mul_30/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �\
mul_30Mulones_like_11:output:0mul_30/y:output:0*
T0*
_output_shapes
:I
Sqrt_3SqrtSelectV2_21:output:0*
T0*
_output_shapes
:=
Log_8Log
Sqrt_3:y:0*
T0*
_output_shapes
:P
truediv_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_8RealDiv	Log_8:y:0truediv_8/y:output:0*
T0*
_output_shapes
:A
Neg_18Negtruediv_8:z:0*
T0*
_output_shapes
:D
Round_8Roundtruediv_8:z:0*
T0*
_output_shapes
:M
add_29AddV2
Neg_18:y:0Round_8:y:0*
T0*
_output_shapes
:P
StopGradient_13StopGradient
add_29:z:0*
T0*
_output_shapes
:]
add_30AddV2truediv_8:z:0StopGradient_13:output:0*
T0*
_output_shapes
:^
clip_by_value_8/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Aw
clip_by_value_8/MinimumMinimum
add_30:z:0"clip_by_value_8/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_8Maximumclip_by_value_8/Minimum:z:0clip_by_value_8/y:output:0*
T0*
_output_shapes
:M
mul_31/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Z
mul_31Mulmul_31/x:output:0clip_by_value_8:z:0*
T0*
_output_shapes
:a
SelectV2_22SelectV2Less_17:z:0
mul_30:z:0
mul_31:z:0*
T0*
_output_shapes
:U
GreaterEqual_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *    z
GreaterEqual_4GreaterEqualmoments/Squeeze_1:output:0GreaterEqual_4/y:output:0*
T0*
_output_shapes
:O
LogicalOr_2/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z`
LogicalOr_2	LogicalOrGreaterEqual_4:z:0LogicalOr_2/y:output:0*
_output_shapes
:L
pow_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_7Powpow_7/x:output:0SelectV2_20:output:0*
T0*
_output_shapes
:L
pow_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_8Powpow_8/x:output:0SelectV2_22:output:0*
T0*
_output_shapes
:=
Neg_19Neg	pow_8:z:0*
T0*
_output_shapes
:d
SelectV2_23SelectV2LogicalOr_2:z:0	pow_7:z:0
Neg_19:y:0*
T0*
_output_shapes
:H
Neg_20NegRelu_6:activations:0*
T0*
_output_shapes
:V
add_31AddV2
Neg_20:y:0SelectV2_23:output:0*
T0*
_output_shapes
:M
mul_32/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
mul_32Mulmul_32/x:output:0
add_31:z:0*
T0*
_output_shapes
:P
StopGradient_14StopGradient
mul_32:z:0*
T0*
_output_shapes
:d
add_32AddV2Relu_6:activations:0StopGradient_14:output:0*
T0*
_output_shapes
:Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<t
AssignMovingAvg/ReadVariableOpReadVariableOpreadvariableop_6_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOpreadvariableop_6_resourceAssignMovingAvg/mul:z:0^Abs_3/ReadVariableOp^AssignMovingAvg/ReadVariableOp^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<{
 AssignMovingAvg_1/ReadVariableOpReadVariableOprelu_3_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOprelu_3_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp_10^ReadVariableOp_9^Relu_3/ReadVariableOp^Relu_4/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:a
batchnorm/addAddV2
add_32:z:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Y
batchnorm/mulMulbatchnorm/Rsqrt:y:0	add_5:z:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������Z
batchnorm/mul_2Mul
add_26:z:0batchnorm/mul:z:0*
T0*
_output_shapes
:Z
batchnorm/subSub
add_10:z:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^Abs_1/ReadVariableOp^Abs_3/ReadVariableOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_10^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8^ReadVariableOp_9^Relu/ReadVariableOp^Relu_1/ReadVariableOp^Relu_3/ReadVariableOp^Relu_4/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2,
Abs_1/ReadVariableOpAbs_1/ReadVariableOp2,
Abs_3/ReadVariableOpAbs_3/ReadVariableOp2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12&
ReadVariableOp_10ReadVariableOp_102$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82$
ReadVariableOp_9ReadVariableOp_92*
Relu/ReadVariableOpRelu/ReadVariableOp2.
Relu_1/ReadVariableOpRelu_1/ReadVariableOp2.
Relu_3/ReadVariableOpRelu_3/ReadVariableOp2.
Relu_4/ReadVariableOpRelu_4/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�%
�
O__inference_simplified_encoder_layer_call_and_return_conditional_losses_8204990

inputs 
dense1_8204374:, 
dense1_8204376: 
bn1_8204669: 
bn1_8204671: 
bn1_8204673: 
bn1_8204675:  
dense2_8204679: 
dense2_8204681:
bn2_8204974:
bn2_8204976:
bn2_8204978:
bn2_8204980: 
z_mean_8204984:
z_mean_8204986:
identity��BN1/StatefulPartitionedCall�BN2/StatefulPartitionedCall�dense1/StatefulPartitionedCall�dense2/StatefulPartitionedCall�z_mean/StatefulPartitionedCall�
dense1/StatefulPartitionedCallStatefulPartitionedCallinputsdense1_8204374dense1_8204376*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_8203214�
BN1/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0bn1_8204669bn1_8204671bn1_8204673bn1_8204675*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_BN1_layer_call_and_return_conditional_losses_8204668�
relu1/PartitionedCallPartitionedCall$BN1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_relu1_layer_call_and_return_conditional_losses_8203720�
dense2/StatefulPartitionedCallStatefulPartitionedCallrelu1/PartitionedCall:output:0dense2_8204679dense2_8204681*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense2_layer_call_and_return_conditional_losses_8203789�
BN2/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0bn2_8204974bn2_8204976bn2_8204978bn2_8204980*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_BN2_layer_call_and_return_conditional_losses_8204973�
relu2/PartitionedCallPartitionedCall$BN2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_relu2_layer_call_and_return_conditional_losses_8204295�
z_mean/StatefulPartitionedCallStatefulPartitionedCallrelu2/PartitionedCall:output:0z_mean_8204984z_mean_8204986*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_z_mean_layer_call_and_return_conditional_losses_8204364v
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BN1/StatefulPartitionedCall^BN2/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������,: : : : : : : : : : : : : : 2:
BN1/StatefulPartitionedCallBN1/StatefulPartitionedCall2:
BN2/StatefulPartitionedCallBN2/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:O K
'
_output_shapes
:���������,
 
_user_specified_nameinputs:'#
!
_user_specified_name	8204374:'#
!
_user_specified_name	8204376:'#
!
_user_specified_name	8204669:'#
!
_user_specified_name	8204671:'#
!
_user_specified_name	8204673:'#
!
_user_specified_name	8204675:'#
!
_user_specified_name	8204679:'#
!
_user_specified_name	8204681:'	#
!
_user_specified_name	8204974:'
#
!
_user_specified_name	8204976:'#
!
_user_specified_name	8204978:'#
!
_user_specified_name	8204980:'#
!
_user_specified_name	8204984:'#
!
_user_specified_name	8204986
��
�
@__inference_BN1_layer_call_and_return_conditional_losses_8203642

inputs%
readvariableop_resource: '
readvariableop_3_resource: '
readvariableop_6_resource: ,
relu_3_readvariableop_resource: 
identity��Abs_1/ReadVariableOp�Abs_3/ReadVariableOp�AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_10�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�ReadVariableOp_6�ReadVariableOp_7�ReadVariableOp_8�ReadVariableOp_9�Relu/ReadVariableOp�Relu_1/ReadVariableOp�Relu_3/ReadVariableOp�Relu_4/ReadVariableOpb
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0P
LessEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ei
	LessEqual	LessEqualReadVariableOp:value:0LessEqual/y:output:0*
T0*
_output_shapes
: g
Relu/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0N
ReluReluRelu/ReadVariableOp:value:0*
T0*
_output_shapes
: l
ones_like/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0i
ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?z
	ones_likeFill(ones_like/Shape/shape_as_tensor:output:0ones_like/Const:output:0*
T0*
_output_shapes
: J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ES
mulMulones_like:output:0mul/y:output:0*
T0*
_output_shapes
: e
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*
_output_shapes
: i
Relu_1/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0R
Relu_1ReluRelu_1/ReadVariableOp:value:0*
T0*
_output_shapes
: K
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3X
LessLessRelu_1:activations:0Less/y:output:0*
T0*
_output_shapes
: Q
SelectV2_1/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3r

SelectV2_1SelectV2Less:z:0SelectV2_1/t:output:0Relu_1:activations:0*
T0*
_output_shapes
: S
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Eo
GreaterEqualGreaterEqualSelectV2_1:output:0GreaterEqual/y:output:0*
T0*
_output_shapes
: k
!ones_like_1/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_1Fill*ones_like_1/Shape/shape_as_tensor:output:0ones_like_1/Const:output:0*
T0*
_output_shapes
: L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   EY
mul_1Mulones_like_1:output:0mul_1/y:output:0*
T0*
_output_shapes
: m

SelectV2_2SelectV2GreaterEqual:z:0	mul_1:z:0SelectV2_1:output:0*
T0*
_output_shapes
: M
Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3\
Less_1LessRelu_1:activations:0Less_1/y:output:0*
T0*
_output_shapes
: k
!ones_like_2/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_2Fill*ones_like_2/Shape/shape_as_tensor:output:0ones_like_2/Const:output:0*
T0*
_output_shapes
: L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �Y
mul_2Mulones_like_2:output:0mul_2/y:output:0*
T0*
_output_shapes
: D
LogLogSelectV2_2:output:0*
T0*
_output_shapes
: N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?T
truedivRealDivLog:y:0truediv/y:output:0*
T0*
_output_shapes
: <
NegNegtruediv:z:0*
T0*
_output_shapes
: @
RoundRoundtruediv:z:0*
T0*
_output_shapes
: E
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes
: J
StopGradientStopGradientadd:z:0*
T0*
_output_shapes
: W
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes
: \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Ar
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
: T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �r
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes
: L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_3Mulmul_3/x:output:0clip_by_value:z:0*
T0*
_output_shapes
: ]

SelectV2_3SelectV2
Less_1:z:0	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: d
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0K
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes
: >
Relu_2Relu	Neg_1:y:0*
T0*
_output_shapes
: L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
mul_4MulRelu_2:activations:0mul_4/y:output:0*
T0*
_output_shapes
: M
Less_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_2Less	mul_4:z:0Less_2/y:output:0*
T0*
_output_shapes
: Q
SelectV2_4/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3i

SelectV2_4SelectV2
Less_2:z:0SelectV2_4/t:output:0	mul_4:z:0*
T0*
_output_shapes
: U
GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Es
GreaterEqual_1GreaterEqualSelectV2_4:output:0GreaterEqual_1/y:output:0*
T0*
_output_shapes
: k
!ones_like_3/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_3Fill*ones_like_3/Shape/shape_as_tensor:output:0ones_like_3/Const:output:0*
T0*
_output_shapes
: L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *   EY
mul_5Mulones_like_3:output:0mul_5/y:output:0*
T0*
_output_shapes
: o

SelectV2_5SelectV2GreaterEqual_1:z:0	mul_5:z:0SelectV2_4:output:0*
T0*
_output_shapes
: M
Less_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_3Less	mul_4:z:0Less_3/y:output:0*
T0*
_output_shapes
: k
!ones_like_4/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_4Fill*ones_like_4/Shape/shape_as_tensor:output:0ones_like_4/Const:output:0*
T0*
_output_shapes
: L
mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �Y
mul_6Mulones_like_4:output:0mul_6/y:output:0*
T0*
_output_shapes
: F
Log_1LogSelectV2_5:output:0*
T0*
_output_shapes
: P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_1RealDiv	Log_1:y:0truediv_1/y:output:0*
T0*
_output_shapes
: @
Neg_2Negtruediv_1:z:0*
T0*
_output_shapes
: D
Round_1Roundtruediv_1:z:0*
T0*
_output_shapes
: K
add_2AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
: N
StopGradient_1StopGradient	add_2:z:0*
T0*
_output_shapes
: [
add_3AddV2truediv_1:z:0StopGradient_1:output:0*
T0*
_output_shapes
: ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Av
clip_by_value_1/MinimumMinimum	add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
: V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
: L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?X
mul_7Mulmul_7/x:output:0clip_by_value_1:z:0*
T0*
_output_shapes
: ]

SelectV2_6SelectV2
Less_3:z:0	mul_6:z:0	mul_7:z:0*
T0*
_output_shapes
: d
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0U
GreaterEqual_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    x
GreaterEqual_2GreaterEqualReadVariableOp_2:value:0GreaterEqual_2/y:output:0*
T0*
_output_shapes
: M
LogicalOr/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z\
	LogicalOr	LogicalOrGreaterEqual_2:z:0LogicalOr/y:output:0*
_output_shapes
: J
pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
powPowpow/x:output:0SelectV2_3:output:0*
T0*
_output_shapes
: L
pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @X
pow_1Powpow_1/x:output:0SelectV2_6:output:0*
T0*
_output_shapes
: <
Neg_3Neg	pow_1:z:0*
T0*
_output_shapes
: ^

SelectV2_7SelectV2LogicalOr:z:0pow:z:0	Neg_3:y:0*
T0*
_output_shapes
: D
Neg_4NegSelectV2:output:0*
T0*
_output_shapes
: S
add_4AddV2	Neg_4:y:0SelectV2_7:output:0*
T0*
_output_shapes
: L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_8Mulmul_8/x:output:0	add_4:z:0*
T0*
_output_shapes
: N
StopGradient_2StopGradient	mul_8:z:0*
T0*
_output_shapes
: _
add_5AddV2SelectV2:output:0StopGradient_2:output:0*
T0*
_output_shapes
: f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
: *
dtype0K
SignSignReadVariableOp_3:value:0*
T0*
_output_shapes
: 9
AbsAbsSign:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?H
subSubsub/x:output:0Abs:y:0*
T0*
_output_shapes
: F
add_6AddV2Sign:y:0sub:z:0*
T0*
_output_shapes
: j
Abs_1/ReadVariableOpReadVariableOpreadvariableop_3_resource*
_output_shapes
: *
dtype0O
Abs_1AbsAbs_1/ReadVariableOp:value:0*
T0*
_output_shapes
: M
Less_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_4Less	Abs_1:y:0Less_4/y:output:0*
T0*
_output_shapes
: Q
SelectV2_8/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3i

SelectV2_8SelectV2
Less_4:z:0SelectV2_8/t:output:0	Abs_1:y:0*
T0*
_output_shapes
: M
Less_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_5Less	Abs_1:y:0Less_5/y:output:0*
T0*
_output_shapes
: k
!ones_like_5/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_5Fill*ones_like_5/Shape/shape_as_tensor:output:0ones_like_5/Const:output:0*
T0*
_output_shapes
: L
mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �Y
mul_9Mulones_like_5:output:0mul_9/y:output:0*
T0*
_output_shapes
: F
Log_2LogSelectV2_8:output:0*
T0*
_output_shapes
: P
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_2RealDiv	Log_2:y:0truediv_2/y:output:0*
T0*
_output_shapes
: @
Neg_5Negtruediv_2:z:0*
T0*
_output_shapes
: D
Round_2Roundtruediv_2:z:0*
T0*
_output_shapes
: K
add_7AddV2	Neg_5:y:0Round_2:y:0*
T0*
_output_shapes
: N
StopGradient_3StopGradient	add_7:z:0*
T0*
_output_shapes
: [
add_8AddV2truediv_2:z:0StopGradient_3:output:0*
T0*
_output_shapes
: ^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@v
clip_by_value_2/MinimumMinimum	add_8:z:0"clip_by_value_2/Minimum/y:output:0*
T0*
_output_shapes
: V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*
_output_shapes
: M
mul_10/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
mul_10Mulmul_10/x:output:0clip_by_value_2:z:0*
T0*
_output_shapes
: ^

SelectV2_9SelectV2
Less_5:z:0	mul_9:z:0
mul_10:z:0*
T0*
_output_shapes
: L
pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @X
pow_2Powpow_2/x:output:0SelectV2_9:output:0*
T0*
_output_shapes
: H
mul_11Mul	add_6:z:0	pow_2:z:0*
T0*
_output_shapes
: f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
: *
dtype0K
Neg_6NegReadVariableOp_4:value:0*
T0*
_output_shapes
: J
add_9AddV2	Neg_6:y:0
mul_11:z:0*
T0*
_output_shapes
: M
mul_12/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?P
mul_12Mulmul_12/x:output:0	add_9:z:0*
T0*
_output_shapes
: O
StopGradient_4StopGradient
mul_12:z:0*
T0*
_output_shapes
: f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
: *
dtype0g
add_10AddV2ReadVariableOp_5:value:0StopGradient_4:output:0*
T0*
_output_shapes
: f
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource*
_output_shapes
: *
dtype0M
Sign_1SignReadVariableOp_6:value:0*
T0*
_output_shapes
: =
Abs_2Abs
Sign_1:y:0*
T0*
_output_shapes
: L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_1Subsub_1/x:output:0	Abs_2:y:0*
T0*
_output_shapes
: K
add_11AddV2
Sign_1:y:0	sub_1:z:0*
T0*
_output_shapes
: j
Abs_3/ReadVariableOpReadVariableOpreadvariableop_6_resource*
_output_shapes
: *
dtype0O
Abs_3AbsAbs_3/ReadVariableOp:value:0*
T0*
_output_shapes
: M
Less_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_6Less	Abs_3:y:0Less_6/y:output:0*
T0*
_output_shapes
: R
SelectV2_10/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3k
SelectV2_10SelectV2
Less_6:z:0SelectV2_10/t:output:0	Abs_3:y:0*
T0*
_output_shapes
: M
Less_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_7Less	Abs_3:y:0Less_7/y:output:0*
T0*
_output_shapes
: k
!ones_like_6/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_6Fill*ones_like_6/Shape/shape_as_tensor:output:0ones_like_6/Const:output:0*
T0*
_output_shapes
: M
mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �[
mul_13Mulones_like_6:output:0mul_13/y:output:0*
T0*
_output_shapes
: G
Log_3LogSelectV2_10:output:0*
T0*
_output_shapes
: P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_3RealDiv	Log_3:y:0truediv_3/y:output:0*
T0*
_output_shapes
: @
Neg_7Negtruediv_3:z:0*
T0*
_output_shapes
: D
Round_3Roundtruediv_3:z:0*
T0*
_output_shapes
: L
add_12AddV2	Neg_7:y:0Round_3:y:0*
T0*
_output_shapes
: O
StopGradient_5StopGradient
add_12:z:0*
T0*
_output_shapes
: \
add_13AddV2truediv_3:z:0StopGradient_5:output:0*
T0*
_output_shapes
: ^
clip_by_value_3/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@w
clip_by_value_3/MinimumMinimum
add_13:z:0"clip_by_value_3/Minimum/y:output:0*
T0*
_output_shapes
: V
clip_by_value_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0*
_output_shapes
: M
mul_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
mul_14Mulmul_14/x:output:0clip_by_value_3:z:0*
T0*
_output_shapes
: `
SelectV2_11SelectV2
Less_7:z:0
mul_13:z:0
mul_14:z:0*
T0*
_output_shapes
: L
pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_3Powpow_3/x:output:0SelectV2_11:output:0*
T0*
_output_shapes
: I
mul_15Mul
add_11:z:0	pow_3:z:0*
T0*
_output_shapes
: f
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource*
_output_shapes
: *
dtype0K
Neg_8NegReadVariableOp_7:value:0*
T0*
_output_shapes
: K
add_14AddV2	Neg_8:y:0
mul_15:z:0*
T0*
_output_shapes
: M
mul_16/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
mul_16Mulmul_16/x:output:0
add_14:z:0*
T0*
_output_shapes
: O
StopGradient_6StopGradient
mul_16:z:0*
T0*
_output_shapes
: f
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource*
_output_shapes
: *
dtype0g
add_15AddV2ReadVariableOp_8:value:0StopGradient_6:output:0*
T0*
_output_shapes
: p
Relu_3/ReadVariableOpReadVariableOprelu_3_readvariableop_resource*
_output_shapes
: *
dtype0R
Relu_3ReluRelu_3/ReadVariableOp:value:0*
T0*
_output_shapes
: p
Relu_4/ReadVariableOpReadVariableOprelu_3_readvariableop_resource*
_output_shapes
: *
dtype0R
Relu_4ReluRelu_4/ReadVariableOp:value:0*
T0*
_output_shapes
: M
Less_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3\
Less_8LessRelu_4:activations:0Less_8/y:output:0*
T0*
_output_shapes
: R
SelectV2_12/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3v
SelectV2_12SelectV2
Less_8:z:0SelectV2_12/t:output:0Relu_4:activations:0*
T0*
_output_shapes
: M
Less_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3\
Less_9LessRelu_4:activations:0Less_9/y:output:0*
T0*
_output_shapes
: k
!ones_like_7/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_7Fill*ones_like_7/Shape/shape_as_tensor:output:0ones_like_7/Const:output:0*
T0*
_output_shapes
: M
mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �[
mul_17Mulones_like_7:output:0mul_17/y:output:0*
T0*
_output_shapes
: G
SqrtSqrtSelectV2_12:output:0*
T0*
_output_shapes
: ;
Log_4LogSqrt:y:0*
T0*
_output_shapes
: P
truediv_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_4RealDiv	Log_4:y:0truediv_4/y:output:0*
T0*
_output_shapes
: @
Neg_9Negtruediv_4:z:0*
T0*
_output_shapes
: D
Round_4Roundtruediv_4:z:0*
T0*
_output_shapes
: L
add_16AddV2	Neg_9:y:0Round_4:y:0*
T0*
_output_shapes
: O
StopGradient_7StopGradient
add_16:z:0*
T0*
_output_shapes
: \
add_17AddV2truediv_4:z:0StopGradient_7:output:0*
T0*
_output_shapes
: ^
clip_by_value_4/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Aw
clip_by_value_4/MinimumMinimum
add_17:z:0"clip_by_value_4/Minimum/y:output:0*
T0*
_output_shapes
: V
clip_by_value_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_4Maximumclip_by_value_4/Minimum:z:0clip_by_value_4/y:output:0*
T0*
_output_shapes
: M
mul_18/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Z
mul_18Mulmul_18/x:output:0clip_by_value_4:z:0*
T0*
_output_shapes
: `
SelectV2_13SelectV2
Less_9:z:0
mul_17:z:0
mul_18:z:0*
T0*
_output_shapes
: k
ReadVariableOp_9ReadVariableOprelu_3_readvariableop_resource*
_output_shapes
: *
dtype0L
Neg_10NegReadVariableOp_9:value:0*
T0*
_output_shapes
: ?
Relu_5Relu
Neg_10:y:0*
T0*
_output_shapes
: M
mul_19/yConst*
_output_shapes
: *
dtype0*
valueB
 *    [
mul_19MulRelu_5:activations:0mul_19/y:output:0*
T0*
_output_shapes
: N
	Less_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3T
Less_10Less
mul_19:z:0Less_10/y:output:0*
T0*
_output_shapes
: R
SelectV2_14/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3m
SelectV2_14SelectV2Less_10:z:0SelectV2_14/t:output:0
mul_19:z:0*
T0*
_output_shapes
: N
	Less_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3T
Less_11Less
mul_19:z:0Less_11/y:output:0*
T0*
_output_shapes
: k
!ones_like_8/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_8Fill*ones_like_8/Shape/shape_as_tensor:output:0ones_like_8/Const:output:0*
T0*
_output_shapes
: M
mul_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �[
mul_20Mulones_like_8:output:0mul_20/y:output:0*
T0*
_output_shapes
: I
Sqrt_1SqrtSelectV2_14:output:0*
T0*
_output_shapes
: =
Log_5Log
Sqrt_1:y:0*
T0*
_output_shapes
: P
truediv_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_5RealDiv	Log_5:y:0truediv_5/y:output:0*
T0*
_output_shapes
: A
Neg_11Negtruediv_5:z:0*
T0*
_output_shapes
: D
Round_5Roundtruediv_5:z:0*
T0*
_output_shapes
: M
add_18AddV2
Neg_11:y:0Round_5:y:0*
T0*
_output_shapes
: O
StopGradient_8StopGradient
add_18:z:0*
T0*
_output_shapes
: \
add_19AddV2truediv_5:z:0StopGradient_8:output:0*
T0*
_output_shapes
: ^
clip_by_value_5/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Aw
clip_by_value_5/MinimumMinimum
add_19:z:0"clip_by_value_5/Minimum/y:output:0*
T0*
_output_shapes
: V
clip_by_value_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_5Maximumclip_by_value_5/Minimum:z:0clip_by_value_5/y:output:0*
T0*
_output_shapes
: M
mul_21/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Z
mul_21Mulmul_21/x:output:0clip_by_value_5:z:0*
T0*
_output_shapes
: a
SelectV2_15SelectV2Less_11:z:0
mul_20:z:0
mul_21:z:0*
T0*
_output_shapes
: l
ReadVariableOp_10ReadVariableOprelu_3_readvariableop_resource*
_output_shapes
: *
dtype0U
GreaterEqual_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *    y
GreaterEqual_3GreaterEqualReadVariableOp_10:value:0GreaterEqual_3/y:output:0*
T0*
_output_shapes
: O
LogicalOr_1/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z`
LogicalOr_1	LogicalOrGreaterEqual_3:z:0LogicalOr_1/y:output:0*
_output_shapes
: L
pow_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_4Powpow_4/x:output:0SelectV2_13:output:0*
T0*
_output_shapes
: L
pow_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_5Powpow_5/x:output:0SelectV2_15:output:0*
T0*
_output_shapes
: =
Neg_12Neg	pow_5:z:0*
T0*
_output_shapes
: d
SelectV2_16SelectV2LogicalOr_1:z:0	pow_4:z:0
Neg_12:y:0*
T0*
_output_shapes
: H
Neg_13NegRelu_3:activations:0*
T0*
_output_shapes
: V
add_20AddV2
Neg_13:y:0SelectV2_16:output:0*
T0*
_output_shapes
: M
mul_22/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
mul_22Mulmul_22/x:output:0
add_20:z:0*
T0*
_output_shapes
: O
StopGradient_9StopGradient
mul_22:z:0*
T0*
_output_shapes
: c
add_21AddV2Relu_3:activations:0StopGradient_9:output:0*
T0*
_output_shapes
: h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

: �
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:��������� l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 M
Sign_2Signmoments/Squeeze:output:0*
T0*
_output_shapes
: =
Abs_4Abs
Sign_2:y:0*
T0*
_output_shapes
: L
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_2Subsub_2/x:output:0	Abs_4:y:0*
T0*
_output_shapes
: K
add_22AddV2
Sign_2:y:0	sub_2:z:0*
T0*
_output_shapes
: K
Abs_5Absmoments/Squeeze:output:0*
T0*
_output_shapes
: N
	Less_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3S
Less_12Less	Abs_5:y:0Less_12/y:output:0*
T0*
_output_shapes
: R
SelectV2_17/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3l
SelectV2_17SelectV2Less_12:z:0SelectV2_17/t:output:0	Abs_5:y:0*
T0*
_output_shapes
: N
	Less_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3S
Less_13Less	Abs_5:y:0Less_13/y:output:0*
T0*
_output_shapes
: k
!ones_like_9/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_9/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_9Fill*ones_like_9/Shape/shape_as_tensor:output:0ones_like_9/Const:output:0*
T0*
_output_shapes
: M
mul_23/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �[
mul_23Mulones_like_9:output:0mul_23/y:output:0*
T0*
_output_shapes
: G
Log_6LogSelectV2_17:output:0*
T0*
_output_shapes
: P
truediv_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_6RealDiv	Log_6:y:0truediv_6/y:output:0*
T0*
_output_shapes
: A
Neg_14Negtruediv_6:z:0*
T0*
_output_shapes
: D
Round_6Roundtruediv_6:z:0*
T0*
_output_shapes
: M
add_23AddV2
Neg_14:y:0Round_6:y:0*
T0*
_output_shapes
: P
StopGradient_10StopGradient
add_23:z:0*
T0*
_output_shapes
: ]
add_24AddV2truediv_6:z:0StopGradient_10:output:0*
T0*
_output_shapes
: ^
clip_by_value_6/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@w
clip_by_value_6/MinimumMinimum
add_24:z:0"clip_by_value_6/Minimum/y:output:0*
T0*
_output_shapes
: V
clip_by_value_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_6Maximumclip_by_value_6/Minimum:z:0clip_by_value_6/y:output:0*
T0*
_output_shapes
: M
mul_24/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
mul_24Mulmul_24/x:output:0clip_by_value_6:z:0*
T0*
_output_shapes
: a
SelectV2_18SelectV2Less_13:z:0
mul_23:z:0
mul_24:z:0*
T0*
_output_shapes
: L
pow_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_6Powpow_6/x:output:0SelectV2_18:output:0*
T0*
_output_shapes
: I
mul_25Mul
add_22:z:0	pow_6:z:0*
T0*
_output_shapes
: L
Neg_15Negmoments/Squeeze:output:0*
T0*
_output_shapes
: L
add_25AddV2
Neg_15:y:0
mul_25:z:0*
T0*
_output_shapes
: M
mul_26/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
mul_26Mulmul_26/x:output:0
add_25:z:0*
T0*
_output_shapes
: P
StopGradient_11StopGradient
mul_26:z:0*
T0*
_output_shapes
: h
add_26AddV2moments/Squeeze:output:0StopGradient_11:output:0*
T0*
_output_shapes
: O
Relu_6Relumoments/Squeeze_1:output:0*
T0*
_output_shapes
: O
Relu_7Relumoments/Squeeze_1:output:0*
T0*
_output_shapes
: N
	Less_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3^
Less_14LessRelu_7:activations:0Less_14/y:output:0*
T0*
_output_shapes
: R
SelectV2_19/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3w
SelectV2_19SelectV2Less_14:z:0SelectV2_19/t:output:0Relu_7:activations:0*
T0*
_output_shapes
: N
	Less_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3^
Less_15LessRelu_7:activations:0Less_15/y:output:0*
T0*
_output_shapes
: l
"ones_like_10/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: W
ones_like_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_10Fill+ones_like_10/Shape/shape_as_tensor:output:0ones_like_10/Const:output:0*
T0*
_output_shapes
: M
mul_27/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �\
mul_27Mulones_like_10:output:0mul_27/y:output:0*
T0*
_output_shapes
: I
Sqrt_2SqrtSelectV2_19:output:0*
T0*
_output_shapes
: =
Log_7Log
Sqrt_2:y:0*
T0*
_output_shapes
: P
truediv_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_7RealDiv	Log_7:y:0truediv_7/y:output:0*
T0*
_output_shapes
: A
Neg_16Negtruediv_7:z:0*
T0*
_output_shapes
: D
Round_7Roundtruediv_7:z:0*
T0*
_output_shapes
: M
add_27AddV2
Neg_16:y:0Round_7:y:0*
T0*
_output_shapes
: P
StopGradient_12StopGradient
add_27:z:0*
T0*
_output_shapes
: ]
add_28AddV2truediv_7:z:0StopGradient_12:output:0*
T0*
_output_shapes
: ^
clip_by_value_7/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Aw
clip_by_value_7/MinimumMinimum
add_28:z:0"clip_by_value_7/Minimum/y:output:0*
T0*
_output_shapes
: V
clip_by_value_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_7Maximumclip_by_value_7/Minimum:z:0clip_by_value_7/y:output:0*
T0*
_output_shapes
: M
mul_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Z
mul_28Mulmul_28/x:output:0clip_by_value_7:z:0*
T0*
_output_shapes
: a
SelectV2_20SelectV2Less_15:z:0
mul_27:z:0
mul_28:z:0*
T0*
_output_shapes
: N
Neg_17Negmoments/Squeeze_1:output:0*
T0*
_output_shapes
: ?
Relu_8Relu
Neg_17:y:0*
T0*
_output_shapes
: M
mul_29/yConst*
_output_shapes
: *
dtype0*
valueB
 *    [
mul_29MulRelu_8:activations:0mul_29/y:output:0*
T0*
_output_shapes
: N
	Less_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3T
Less_16Less
mul_29:z:0Less_16/y:output:0*
T0*
_output_shapes
: R
SelectV2_21/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3m
SelectV2_21SelectV2Less_16:z:0SelectV2_21/t:output:0
mul_29:z:0*
T0*
_output_shapes
: N
	Less_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3T
Less_17Less
mul_29:z:0Less_17/y:output:0*
T0*
_output_shapes
: l
"ones_like_11/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: W
ones_like_11/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_11Fill+ones_like_11/Shape/shape_as_tensor:output:0ones_like_11/Const:output:0*
T0*
_output_shapes
: M
mul_30/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �\
mul_30Mulones_like_11:output:0mul_30/y:output:0*
T0*
_output_shapes
: I
Sqrt_3SqrtSelectV2_21:output:0*
T0*
_output_shapes
: =
Log_8Log
Sqrt_3:y:0*
T0*
_output_shapes
: P
truediv_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_8RealDiv	Log_8:y:0truediv_8/y:output:0*
T0*
_output_shapes
: A
Neg_18Negtruediv_8:z:0*
T0*
_output_shapes
: D
Round_8Roundtruediv_8:z:0*
T0*
_output_shapes
: M
add_29AddV2
Neg_18:y:0Round_8:y:0*
T0*
_output_shapes
: P
StopGradient_13StopGradient
add_29:z:0*
T0*
_output_shapes
: ]
add_30AddV2truediv_8:z:0StopGradient_13:output:0*
T0*
_output_shapes
: ^
clip_by_value_8/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Aw
clip_by_value_8/MinimumMinimum
add_30:z:0"clip_by_value_8/Minimum/y:output:0*
T0*
_output_shapes
: V
clip_by_value_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_8Maximumclip_by_value_8/Minimum:z:0clip_by_value_8/y:output:0*
T0*
_output_shapes
: M
mul_31/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Z
mul_31Mulmul_31/x:output:0clip_by_value_8:z:0*
T0*
_output_shapes
: a
SelectV2_22SelectV2Less_17:z:0
mul_30:z:0
mul_31:z:0*
T0*
_output_shapes
: U
GreaterEqual_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *    z
GreaterEqual_4GreaterEqualmoments/Squeeze_1:output:0GreaterEqual_4/y:output:0*
T0*
_output_shapes
: O
LogicalOr_2/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z`
LogicalOr_2	LogicalOrGreaterEqual_4:z:0LogicalOr_2/y:output:0*
_output_shapes
: L
pow_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_7Powpow_7/x:output:0SelectV2_20:output:0*
T0*
_output_shapes
: L
pow_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_8Powpow_8/x:output:0SelectV2_22:output:0*
T0*
_output_shapes
: =
Neg_19Neg	pow_8:z:0*
T0*
_output_shapes
: d
SelectV2_23SelectV2LogicalOr_2:z:0	pow_7:z:0
Neg_19:y:0*
T0*
_output_shapes
: H
Neg_20NegRelu_6:activations:0*
T0*
_output_shapes
: V
add_31AddV2
Neg_20:y:0SelectV2_23:output:0*
T0*
_output_shapes
: M
mul_32/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
mul_32Mulmul_32/x:output:0
add_31:z:0*
T0*
_output_shapes
: P
StopGradient_14StopGradient
mul_32:z:0*
T0*
_output_shapes
: d
add_32AddV2Relu_6:activations:0StopGradient_14:output:0*
T0*
_output_shapes
: Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<t
AssignMovingAvg/ReadVariableOpReadVariableOpreadvariableop_6_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvgAssignSubVariableOpreadvariableop_6_resourceAssignMovingAvg/mul:z:0^Abs_3/ReadVariableOp^AssignMovingAvg/ReadVariableOp^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<{
 AssignMovingAvg_1/ReadVariableOpReadVariableOprelu_3_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: ~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvg_1AssignSubVariableOprelu_3_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp_10^ReadVariableOp_9^Relu_3/ReadVariableOp^Relu_4/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:a
batchnorm/addAddV2
add_32:z:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: Y
batchnorm/mulMulbatchnorm/Rsqrt:y:0	add_5:z:0*
T0*
_output_shapes
: c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:��������� Z
batchnorm/mul_2Mul
add_26:z:0batchnorm/mul:z:0*
T0*
_output_shapes
: Z
batchnorm/subSub
add_10:z:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^Abs_1/ReadVariableOp^Abs_3/ReadVariableOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_10^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8^ReadVariableOp_9^Relu/ReadVariableOp^Relu_1/ReadVariableOp^Relu_3/ReadVariableOp^Relu_4/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 2,
Abs_1/ReadVariableOpAbs_1/ReadVariableOp2,
Abs_3/ReadVariableOpAbs_3/ReadVariableOp2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12&
ReadVariableOp_10ReadVariableOp_102$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82$
ReadVariableOp_9ReadVariableOp_92*
Relu/ReadVariableOpRelu/ReadVariableOp2.
Relu_1/ReadVariableOpRelu_1/ReadVariableOp2.
Relu_3/ReadVariableOpRelu_3/ReadVariableOp2.
Relu_4/ReadVariableOpRelu_4/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�	
�
%__inference_BN2_layer_call_fn_8206159

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_BN2_layer_call_and_return_conditional_losses_8204217o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:'#
!
_user_specified_name	8206149:'#
!
_user_specified_name	8206151:'#
!
_user_specified_name	8206153:'#
!
_user_specified_name	8206155
��
�
@__inference_BN2_layer_call_and_return_conditional_losses_8206885

inputs%
readvariableop_resource:'
readvariableop_3_resource:'
readvariableop_6_resource:,
relu_3_readvariableop_resource:
identity��Abs_1/ReadVariableOp�Abs_3/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_10�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�ReadVariableOp_6�ReadVariableOp_7�ReadVariableOp_8�ReadVariableOp_9�Relu/ReadVariableOp�Relu_1/ReadVariableOp�Relu_3/ReadVariableOp�Relu_4/ReadVariableOpb
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0P
LessEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ei
	LessEqual	LessEqualReadVariableOp:value:0LessEqual/y:output:0*
T0*
_output_shapes
:g
Relu/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0N
ReluReluRelu/ReadVariableOp:value:0*
T0*
_output_shapes
:l
ones_like/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0i
ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?z
	ones_likeFill(ones_like/Shape/shape_as_tensor:output:0ones_like/Const:output:0*
T0*
_output_shapes
:J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ES
mulMulones_like:output:0mul/y:output:0*
T0*
_output_shapes
:e
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*
_output_shapes
:i
Relu_1/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0R
Relu_1ReluRelu_1/ReadVariableOp:value:0*
T0*
_output_shapes
:K
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3X
LessLessRelu_1:activations:0Less/y:output:0*
T0*
_output_shapes
:Q
SelectV2_1/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3r

SelectV2_1SelectV2Less:z:0SelectV2_1/t:output:0Relu_1:activations:0*
T0*
_output_shapes
:S
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Eo
GreaterEqualGreaterEqualSelectV2_1:output:0GreaterEqual/y:output:0*
T0*
_output_shapes
:k
!ones_like_1/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_1Fill*ones_like_1/Shape/shape_as_tensor:output:0ones_like_1/Const:output:0*
T0*
_output_shapes
:L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   EY
mul_1Mulones_like_1:output:0mul_1/y:output:0*
T0*
_output_shapes
:m

SelectV2_2SelectV2GreaterEqual:z:0	mul_1:z:0SelectV2_1:output:0*
T0*
_output_shapes
:M
Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3\
Less_1LessRelu_1:activations:0Less_1/y:output:0*
T0*
_output_shapes
:k
!ones_like_2/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_2Fill*ones_like_2/Shape/shape_as_tensor:output:0ones_like_2/Const:output:0*
T0*
_output_shapes
:L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �Y
mul_2Mulones_like_2:output:0mul_2/y:output:0*
T0*
_output_shapes
:D
LogLogSelectV2_2:output:0*
T0*
_output_shapes
:N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?T
truedivRealDivLog:y:0truediv/y:output:0*
T0*
_output_shapes
:<
NegNegtruediv:z:0*
T0*
_output_shapes
:@
RoundRoundtruediv:z:0*
T0*
_output_shapes
:E
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes
:J
StopGradientStopGradientadd:z:0*
T0*
_output_shapes
:W
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes
:\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Ar
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �r
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes
:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_3Mulmul_3/x:output:0clip_by_value:z:0*
T0*
_output_shapes
:]

SelectV2_3SelectV2
Less_1:z:0	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:d
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0K
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes
:>
Relu_2Relu	Neg_1:y:0*
T0*
_output_shapes
:L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
mul_4MulRelu_2:activations:0mul_4/y:output:0*
T0*
_output_shapes
:M
Less_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_2Less	mul_4:z:0Less_2/y:output:0*
T0*
_output_shapes
:Q
SelectV2_4/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3i

SelectV2_4SelectV2
Less_2:z:0SelectV2_4/t:output:0	mul_4:z:0*
T0*
_output_shapes
:U
GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Es
GreaterEqual_1GreaterEqualSelectV2_4:output:0GreaterEqual_1/y:output:0*
T0*
_output_shapes
:k
!ones_like_3/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_3Fill*ones_like_3/Shape/shape_as_tensor:output:0ones_like_3/Const:output:0*
T0*
_output_shapes
:L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *   EY
mul_5Mulones_like_3:output:0mul_5/y:output:0*
T0*
_output_shapes
:o

SelectV2_5SelectV2GreaterEqual_1:z:0	mul_5:z:0SelectV2_4:output:0*
T0*
_output_shapes
:M
Less_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_3Less	mul_4:z:0Less_3/y:output:0*
T0*
_output_shapes
:k
!ones_like_4/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_4Fill*ones_like_4/Shape/shape_as_tensor:output:0ones_like_4/Const:output:0*
T0*
_output_shapes
:L
mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �Y
mul_6Mulones_like_4:output:0mul_6/y:output:0*
T0*
_output_shapes
:F
Log_1LogSelectV2_5:output:0*
T0*
_output_shapes
:P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_1RealDiv	Log_1:y:0truediv_1/y:output:0*
T0*
_output_shapes
:@
Neg_2Negtruediv_1:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_1:z:0*
T0*
_output_shapes
:K
add_2AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_1StopGradient	add_2:z:0*
T0*
_output_shapes
:[
add_3AddV2truediv_1:z:0StopGradient_1:output:0*
T0*
_output_shapes
:^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Av
clip_by_value_1/MinimumMinimum	add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?X
mul_7Mulmul_7/x:output:0clip_by_value_1:z:0*
T0*
_output_shapes
:]

SelectV2_6SelectV2
Less_3:z:0	mul_6:z:0	mul_7:z:0*
T0*
_output_shapes
:d
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0U
GreaterEqual_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    x
GreaterEqual_2GreaterEqualReadVariableOp_2:value:0GreaterEqual_2/y:output:0*
T0*
_output_shapes
:M
LogicalOr/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z\
	LogicalOr	LogicalOrGreaterEqual_2:z:0LogicalOr/y:output:0*
_output_shapes
:J
pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
powPowpow/x:output:0SelectV2_3:output:0*
T0*
_output_shapes
:L
pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @X
pow_1Powpow_1/x:output:0SelectV2_6:output:0*
T0*
_output_shapes
:<
Neg_3Neg	pow_1:z:0*
T0*
_output_shapes
:^

SelectV2_7SelectV2LogicalOr:z:0pow:z:0	Neg_3:y:0*
T0*
_output_shapes
:D
Neg_4NegSelectV2:output:0*
T0*
_output_shapes
:S
add_4AddV2	Neg_4:y:0SelectV2_7:output:0*
T0*
_output_shapes
:L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_8Mulmul_8/x:output:0	add_4:z:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	mul_8:z:0*
T0*
_output_shapes
:_
add_5AddV2SelectV2:output:0StopGradient_2:output:0*
T0*
_output_shapes
:f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0K
SignSignReadVariableOp_3:value:0*
T0*
_output_shapes
:9
AbsAbsSign:y:0*
T0*
_output_shapes
:J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?H
subSubsub/x:output:0Abs:y:0*
T0*
_output_shapes
:F
add_6AddV2Sign:y:0sub:z:0*
T0*
_output_shapes
:j
Abs_1/ReadVariableOpReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0O
Abs_1AbsAbs_1/ReadVariableOp:value:0*
T0*
_output_shapes
:M
Less_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_4Less	Abs_1:y:0Less_4/y:output:0*
T0*
_output_shapes
:Q
SelectV2_8/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3i

SelectV2_8SelectV2
Less_4:z:0SelectV2_8/t:output:0	Abs_1:y:0*
T0*
_output_shapes
:M
Less_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_5Less	Abs_1:y:0Less_5/y:output:0*
T0*
_output_shapes
:k
!ones_like_5/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_5Fill*ones_like_5/Shape/shape_as_tensor:output:0ones_like_5/Const:output:0*
T0*
_output_shapes
:L
mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �Y
mul_9Mulones_like_5:output:0mul_9/y:output:0*
T0*
_output_shapes
:F
Log_2LogSelectV2_8:output:0*
T0*
_output_shapes
:P
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_2RealDiv	Log_2:y:0truediv_2/y:output:0*
T0*
_output_shapes
:@
Neg_5Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_2Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_7AddV2	Neg_5:y:0Round_2:y:0*
T0*
_output_shapes
:N
StopGradient_3StopGradient	add_7:z:0*
T0*
_output_shapes
:[
add_8AddV2truediv_2:z:0StopGradient_3:output:0*
T0*
_output_shapes
:^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@v
clip_by_value_2/MinimumMinimum	add_8:z:0"clip_by_value_2/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*
_output_shapes
:M
mul_10/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
mul_10Mulmul_10/x:output:0clip_by_value_2:z:0*
T0*
_output_shapes
:^

SelectV2_9SelectV2
Less_5:z:0	mul_9:z:0
mul_10:z:0*
T0*
_output_shapes
:L
pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @X
pow_2Powpow_2/x:output:0SelectV2_9:output:0*
T0*
_output_shapes
:H
mul_11Mul	add_6:z:0	pow_2:z:0*
T0*
_output_shapes
:f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0K
Neg_6NegReadVariableOp_4:value:0*
T0*
_output_shapes
:J
add_9AddV2	Neg_6:y:0
mul_11:z:0*
T0*
_output_shapes
:M
mul_12/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?P
mul_12Mulmul_12/x:output:0	add_9:z:0*
T0*
_output_shapes
:O
StopGradient_4StopGradient
mul_12:z:0*
T0*
_output_shapes
:f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0g
add_10AddV2ReadVariableOp_5:value:0StopGradient_4:output:0*
T0*
_output_shapes
:f
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource*
_output_shapes
:*
dtype0M
Sign_1SignReadVariableOp_6:value:0*
T0*
_output_shapes
:=
Abs_2Abs
Sign_1:y:0*
T0*
_output_shapes
:L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_1Subsub_1/x:output:0	Abs_2:y:0*
T0*
_output_shapes
:K
add_11AddV2
Sign_1:y:0	sub_1:z:0*
T0*
_output_shapes
:j
Abs_3/ReadVariableOpReadVariableOpreadvariableop_6_resource*
_output_shapes
:*
dtype0O
Abs_3AbsAbs_3/ReadVariableOp:value:0*
T0*
_output_shapes
:M
Less_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_6Less	Abs_3:y:0Less_6/y:output:0*
T0*
_output_shapes
:R
SelectV2_10/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3k
SelectV2_10SelectV2
Less_6:z:0SelectV2_10/t:output:0	Abs_3:y:0*
T0*
_output_shapes
:M
Less_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_7Less	Abs_3:y:0Less_7/y:output:0*
T0*
_output_shapes
:k
!ones_like_6/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_6Fill*ones_like_6/Shape/shape_as_tensor:output:0ones_like_6/Const:output:0*
T0*
_output_shapes
:M
mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �[
mul_13Mulones_like_6:output:0mul_13/y:output:0*
T0*
_output_shapes
:G
Log_3LogSelectV2_10:output:0*
T0*
_output_shapes
:P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_3RealDiv	Log_3:y:0truediv_3/y:output:0*
T0*
_output_shapes
:@
Neg_7Negtruediv_3:z:0*
T0*
_output_shapes
:D
Round_3Roundtruediv_3:z:0*
T0*
_output_shapes
:L
add_12AddV2	Neg_7:y:0Round_3:y:0*
T0*
_output_shapes
:O
StopGradient_5StopGradient
add_12:z:0*
T0*
_output_shapes
:\
add_13AddV2truediv_3:z:0StopGradient_5:output:0*
T0*
_output_shapes
:^
clip_by_value_3/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@w
clip_by_value_3/MinimumMinimum
add_13:z:0"clip_by_value_3/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0*
_output_shapes
:M
mul_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
mul_14Mulmul_14/x:output:0clip_by_value_3:z:0*
T0*
_output_shapes
:`
SelectV2_11SelectV2
Less_7:z:0
mul_13:z:0
mul_14:z:0*
T0*
_output_shapes
:L
pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_3Powpow_3/x:output:0SelectV2_11:output:0*
T0*
_output_shapes
:I
mul_15Mul
add_11:z:0	pow_3:z:0*
T0*
_output_shapes
:f
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource*
_output_shapes
:*
dtype0K
Neg_8NegReadVariableOp_7:value:0*
T0*
_output_shapes
:K
add_14AddV2	Neg_8:y:0
mul_15:z:0*
T0*
_output_shapes
:M
mul_16/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
mul_16Mulmul_16/x:output:0
add_14:z:0*
T0*
_output_shapes
:O
StopGradient_6StopGradient
mul_16:z:0*
T0*
_output_shapes
:f
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource*
_output_shapes
:*
dtype0g
add_15AddV2ReadVariableOp_8:value:0StopGradient_6:output:0*
T0*
_output_shapes
:p
Relu_3/ReadVariableOpReadVariableOprelu_3_readvariableop_resource*
_output_shapes
:*
dtype0R
Relu_3ReluRelu_3/ReadVariableOp:value:0*
T0*
_output_shapes
:p
Relu_4/ReadVariableOpReadVariableOprelu_3_readvariableop_resource*
_output_shapes
:*
dtype0R
Relu_4ReluRelu_4/ReadVariableOp:value:0*
T0*
_output_shapes
:M
Less_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3\
Less_8LessRelu_4:activations:0Less_8/y:output:0*
T0*
_output_shapes
:R
SelectV2_12/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3v
SelectV2_12SelectV2
Less_8:z:0SelectV2_12/t:output:0Relu_4:activations:0*
T0*
_output_shapes
:M
Less_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3\
Less_9LessRelu_4:activations:0Less_9/y:output:0*
T0*
_output_shapes
:k
!ones_like_7/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_7Fill*ones_like_7/Shape/shape_as_tensor:output:0ones_like_7/Const:output:0*
T0*
_output_shapes
:M
mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �[
mul_17Mulones_like_7:output:0mul_17/y:output:0*
T0*
_output_shapes
:G
SqrtSqrtSelectV2_12:output:0*
T0*
_output_shapes
:;
Log_4LogSqrt:y:0*
T0*
_output_shapes
:P
truediv_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_4RealDiv	Log_4:y:0truediv_4/y:output:0*
T0*
_output_shapes
:@
Neg_9Negtruediv_4:z:0*
T0*
_output_shapes
:D
Round_4Roundtruediv_4:z:0*
T0*
_output_shapes
:L
add_16AddV2	Neg_9:y:0Round_4:y:0*
T0*
_output_shapes
:O
StopGradient_7StopGradient
add_16:z:0*
T0*
_output_shapes
:\
add_17AddV2truediv_4:z:0StopGradient_7:output:0*
T0*
_output_shapes
:^
clip_by_value_4/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Aw
clip_by_value_4/MinimumMinimum
add_17:z:0"clip_by_value_4/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_4Maximumclip_by_value_4/Minimum:z:0clip_by_value_4/y:output:0*
T0*
_output_shapes
:M
mul_18/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Z
mul_18Mulmul_18/x:output:0clip_by_value_4:z:0*
T0*
_output_shapes
:`
SelectV2_13SelectV2
Less_9:z:0
mul_17:z:0
mul_18:z:0*
T0*
_output_shapes
:k
ReadVariableOp_9ReadVariableOprelu_3_readvariableop_resource*
_output_shapes
:*
dtype0L
Neg_10NegReadVariableOp_9:value:0*
T0*
_output_shapes
:?
Relu_5Relu
Neg_10:y:0*
T0*
_output_shapes
:M
mul_19/yConst*
_output_shapes
: *
dtype0*
valueB
 *    [
mul_19MulRelu_5:activations:0mul_19/y:output:0*
T0*
_output_shapes
:N
	Less_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3T
Less_10Less
mul_19:z:0Less_10/y:output:0*
T0*
_output_shapes
:R
SelectV2_14/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3m
SelectV2_14SelectV2Less_10:z:0SelectV2_14/t:output:0
mul_19:z:0*
T0*
_output_shapes
:N
	Less_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3T
Less_11Less
mul_19:z:0Less_11/y:output:0*
T0*
_output_shapes
:k
!ones_like_8/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_8Fill*ones_like_8/Shape/shape_as_tensor:output:0ones_like_8/Const:output:0*
T0*
_output_shapes
:M
mul_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �[
mul_20Mulones_like_8:output:0mul_20/y:output:0*
T0*
_output_shapes
:I
Sqrt_1SqrtSelectV2_14:output:0*
T0*
_output_shapes
:=
Log_5Log
Sqrt_1:y:0*
T0*
_output_shapes
:P
truediv_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_5RealDiv	Log_5:y:0truediv_5/y:output:0*
T0*
_output_shapes
:A
Neg_11Negtruediv_5:z:0*
T0*
_output_shapes
:D
Round_5Roundtruediv_5:z:0*
T0*
_output_shapes
:M
add_18AddV2
Neg_11:y:0Round_5:y:0*
T0*
_output_shapes
:O
StopGradient_8StopGradient
add_18:z:0*
T0*
_output_shapes
:\
add_19AddV2truediv_5:z:0StopGradient_8:output:0*
T0*
_output_shapes
:^
clip_by_value_5/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Aw
clip_by_value_5/MinimumMinimum
add_19:z:0"clip_by_value_5/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_5Maximumclip_by_value_5/Minimum:z:0clip_by_value_5/y:output:0*
T0*
_output_shapes
:M
mul_21/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Z
mul_21Mulmul_21/x:output:0clip_by_value_5:z:0*
T0*
_output_shapes
:a
SelectV2_15SelectV2Less_11:z:0
mul_20:z:0
mul_21:z:0*
T0*
_output_shapes
:l
ReadVariableOp_10ReadVariableOprelu_3_readvariableop_resource*
_output_shapes
:*
dtype0U
GreaterEqual_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *    y
GreaterEqual_3GreaterEqualReadVariableOp_10:value:0GreaterEqual_3/y:output:0*
T0*
_output_shapes
:O
LogicalOr_1/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z`
LogicalOr_1	LogicalOrGreaterEqual_3:z:0LogicalOr_1/y:output:0*
_output_shapes
:L
pow_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_4Powpow_4/x:output:0SelectV2_13:output:0*
T0*
_output_shapes
:L
pow_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_5Powpow_5/x:output:0SelectV2_15:output:0*
T0*
_output_shapes
:=
Neg_12Neg	pow_5:z:0*
T0*
_output_shapes
:d
SelectV2_16SelectV2LogicalOr_1:z:0	pow_4:z:0
Neg_12:y:0*
T0*
_output_shapes
:H
Neg_13NegRelu_3:activations:0*
T0*
_output_shapes
:V
add_20AddV2
Neg_13:y:0SelectV2_16:output:0*
T0*
_output_shapes
:M
mul_22/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
mul_22Mulmul_22/x:output:0
add_20:z:0*
T0*
_output_shapes
:O
StopGradient_9StopGradient
mul_22:z:0*
T0*
_output_shapes
:c
add_21AddV2Relu_3:activations:0StopGradient_9:output:0*
T0*
_output_shapes
:T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:a
batchnorm/addAddV2
add_21:z:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Y
batchnorm/mulMulbatchnorm/Rsqrt:y:0	add_5:z:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������Z
batchnorm/mul_2Mul
add_15:z:0batchnorm/mul:z:0*
T0*
_output_shapes
:Z
batchnorm/subSub
add_10:z:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^Abs_1/ReadVariableOp^Abs_3/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_10^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8^ReadVariableOp_9^Relu/ReadVariableOp^Relu_1/ReadVariableOp^Relu_3/ReadVariableOp^Relu_4/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2,
Abs_1/ReadVariableOpAbs_1/ReadVariableOp2,
Abs_3/ReadVariableOpAbs_3/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12&
ReadVariableOp_10ReadVariableOp_102$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82$
ReadVariableOp_9ReadVariableOp_92*
Relu/ReadVariableOpRelu/ReadVariableOp2.
Relu_1/ReadVariableOpRelu_1/ReadVariableOp2.
Relu_3/ReadVariableOpRelu_3/ReadVariableOp2.
Relu_4/ReadVariableOpRelu_4/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
(__inference_dense2_layer_call_fn_8206078

inputs
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense2_layer_call_and_return_conditional_losses_8203789o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:'#
!
_user_specified_name	8206072:'#
!
_user_specified_name	8206074
�%
�
O__inference_simplified_encoder_layer_call_and_return_conditional_losses_8204371

inputs 
dense1_8203215:, 
dense1_8203217: 
bn1_8203643: 
bn1_8203645: 
bn1_8203647: 
bn1_8203649:  
dense2_8203790: 
dense2_8203792:
bn2_8204218:
bn2_8204220:
bn2_8204222:
bn2_8204224: 
z_mean_8204365:
z_mean_8204367:
identity��BN1/StatefulPartitionedCall�BN2/StatefulPartitionedCall�dense1/StatefulPartitionedCall�dense2/StatefulPartitionedCall�z_mean/StatefulPartitionedCall�
dense1/StatefulPartitionedCallStatefulPartitionedCallinputsdense1_8203215dense1_8203217*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_8203214�
BN1/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0bn1_8203643bn1_8203645bn1_8203647bn1_8203649*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_BN1_layer_call_and_return_conditional_losses_8203642�
relu1/PartitionedCallPartitionedCall$BN1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_relu1_layer_call_and_return_conditional_losses_8203720�
dense2/StatefulPartitionedCallStatefulPartitionedCallrelu1/PartitionedCall:output:0dense2_8203790dense2_8203792*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense2_layer_call_and_return_conditional_losses_8203789�
BN2/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0bn2_8204218bn2_8204220bn2_8204222bn2_8204224*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_BN2_layer_call_and_return_conditional_losses_8204217�
relu2/PartitionedCallPartitionedCall$BN2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_relu2_layer_call_and_return_conditional_losses_8204295�
z_mean/StatefulPartitionedCallStatefulPartitionedCallrelu2/PartitionedCall:output:0z_mean_8204365z_mean_8204367*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_z_mean_layer_call_and_return_conditional_losses_8204364v
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BN1/StatefulPartitionedCall^BN2/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������,: : : : : : : : : : : : : : 2:
BN1/StatefulPartitionedCallBN1/StatefulPartitionedCall2:
BN2/StatefulPartitionedCallBN2/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:O K
'
_output_shapes
:���������,
 
_user_specified_nameinputs:'#
!
_user_specified_name	8203215:'#
!
_user_specified_name	8203217:'#
!
_user_specified_name	8203643:'#
!
_user_specified_name	8203645:'#
!
_user_specified_name	8203647:'#
!
_user_specified_name	8203649:'#
!
_user_specified_name	8203790:'#
!
_user_specified_name	8203792:'	#
!
_user_specified_name	8204218:'
#
!
_user_specified_name	8204220:'#
!
_user_specified_name	8204222:'#
!
_user_specified_name	8204224:'#
!
_user_specified_name	8204365:'#
!
_user_specified_name	8204367
��
�
@__inference_BN1_layer_call_and_return_conditional_losses_8205995

inputs%
readvariableop_resource: '
readvariableop_3_resource: '
readvariableop_6_resource: ,
relu_3_readvariableop_resource: 
identity��Abs_1/ReadVariableOp�Abs_3/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_10�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�ReadVariableOp_6�ReadVariableOp_7�ReadVariableOp_8�ReadVariableOp_9�Relu/ReadVariableOp�Relu_1/ReadVariableOp�Relu_3/ReadVariableOp�Relu_4/ReadVariableOpb
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0P
LessEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ei
	LessEqual	LessEqualReadVariableOp:value:0LessEqual/y:output:0*
T0*
_output_shapes
: g
Relu/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0N
ReluReluRelu/ReadVariableOp:value:0*
T0*
_output_shapes
: l
ones_like/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0i
ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?z
	ones_likeFill(ones_like/Shape/shape_as_tensor:output:0ones_like/Const:output:0*
T0*
_output_shapes
: J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ES
mulMulones_like:output:0mul/y:output:0*
T0*
_output_shapes
: e
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*
_output_shapes
: i
Relu_1/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0R
Relu_1ReluRelu_1/ReadVariableOp:value:0*
T0*
_output_shapes
: K
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3X
LessLessRelu_1:activations:0Less/y:output:0*
T0*
_output_shapes
: Q
SelectV2_1/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3r

SelectV2_1SelectV2Less:z:0SelectV2_1/t:output:0Relu_1:activations:0*
T0*
_output_shapes
: S
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Eo
GreaterEqualGreaterEqualSelectV2_1:output:0GreaterEqual/y:output:0*
T0*
_output_shapes
: k
!ones_like_1/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_1Fill*ones_like_1/Shape/shape_as_tensor:output:0ones_like_1/Const:output:0*
T0*
_output_shapes
: L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   EY
mul_1Mulones_like_1:output:0mul_1/y:output:0*
T0*
_output_shapes
: m

SelectV2_2SelectV2GreaterEqual:z:0	mul_1:z:0SelectV2_1:output:0*
T0*
_output_shapes
: M
Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3\
Less_1LessRelu_1:activations:0Less_1/y:output:0*
T0*
_output_shapes
: k
!ones_like_2/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_2Fill*ones_like_2/Shape/shape_as_tensor:output:0ones_like_2/Const:output:0*
T0*
_output_shapes
: L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �Y
mul_2Mulones_like_2:output:0mul_2/y:output:0*
T0*
_output_shapes
: D
LogLogSelectV2_2:output:0*
T0*
_output_shapes
: N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?T
truedivRealDivLog:y:0truediv/y:output:0*
T0*
_output_shapes
: <
NegNegtruediv:z:0*
T0*
_output_shapes
: @
RoundRoundtruediv:z:0*
T0*
_output_shapes
: E
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes
: J
StopGradientStopGradientadd:z:0*
T0*
_output_shapes
: W
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes
: \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Ar
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
: T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �r
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes
: L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_3Mulmul_3/x:output:0clip_by_value:z:0*
T0*
_output_shapes
: ]

SelectV2_3SelectV2
Less_1:z:0	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: d
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0K
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes
: >
Relu_2Relu	Neg_1:y:0*
T0*
_output_shapes
: L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
mul_4MulRelu_2:activations:0mul_4/y:output:0*
T0*
_output_shapes
: M
Less_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_2Less	mul_4:z:0Less_2/y:output:0*
T0*
_output_shapes
: Q
SelectV2_4/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3i

SelectV2_4SelectV2
Less_2:z:0SelectV2_4/t:output:0	mul_4:z:0*
T0*
_output_shapes
: U
GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Es
GreaterEqual_1GreaterEqualSelectV2_4:output:0GreaterEqual_1/y:output:0*
T0*
_output_shapes
: k
!ones_like_3/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_3Fill*ones_like_3/Shape/shape_as_tensor:output:0ones_like_3/Const:output:0*
T0*
_output_shapes
: L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *   EY
mul_5Mulones_like_3:output:0mul_5/y:output:0*
T0*
_output_shapes
: o

SelectV2_5SelectV2GreaterEqual_1:z:0	mul_5:z:0SelectV2_4:output:0*
T0*
_output_shapes
: M
Less_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_3Less	mul_4:z:0Less_3/y:output:0*
T0*
_output_shapes
: k
!ones_like_4/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_4Fill*ones_like_4/Shape/shape_as_tensor:output:0ones_like_4/Const:output:0*
T0*
_output_shapes
: L
mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �Y
mul_6Mulones_like_4:output:0mul_6/y:output:0*
T0*
_output_shapes
: F
Log_1LogSelectV2_5:output:0*
T0*
_output_shapes
: P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_1RealDiv	Log_1:y:0truediv_1/y:output:0*
T0*
_output_shapes
: @
Neg_2Negtruediv_1:z:0*
T0*
_output_shapes
: D
Round_1Roundtruediv_1:z:0*
T0*
_output_shapes
: K
add_2AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
: N
StopGradient_1StopGradient	add_2:z:0*
T0*
_output_shapes
: [
add_3AddV2truediv_1:z:0StopGradient_1:output:0*
T0*
_output_shapes
: ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Av
clip_by_value_1/MinimumMinimum	add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
: V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
: L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?X
mul_7Mulmul_7/x:output:0clip_by_value_1:z:0*
T0*
_output_shapes
: ]

SelectV2_6SelectV2
Less_3:z:0	mul_6:z:0	mul_7:z:0*
T0*
_output_shapes
: d
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0U
GreaterEqual_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    x
GreaterEqual_2GreaterEqualReadVariableOp_2:value:0GreaterEqual_2/y:output:0*
T0*
_output_shapes
: M
LogicalOr/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z\
	LogicalOr	LogicalOrGreaterEqual_2:z:0LogicalOr/y:output:0*
_output_shapes
: J
pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
powPowpow/x:output:0SelectV2_3:output:0*
T0*
_output_shapes
: L
pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @X
pow_1Powpow_1/x:output:0SelectV2_6:output:0*
T0*
_output_shapes
: <
Neg_3Neg	pow_1:z:0*
T0*
_output_shapes
: ^

SelectV2_7SelectV2LogicalOr:z:0pow:z:0	Neg_3:y:0*
T0*
_output_shapes
: D
Neg_4NegSelectV2:output:0*
T0*
_output_shapes
: S
add_4AddV2	Neg_4:y:0SelectV2_7:output:0*
T0*
_output_shapes
: L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_8Mulmul_8/x:output:0	add_4:z:0*
T0*
_output_shapes
: N
StopGradient_2StopGradient	mul_8:z:0*
T0*
_output_shapes
: _
add_5AddV2SelectV2:output:0StopGradient_2:output:0*
T0*
_output_shapes
: f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
: *
dtype0K
SignSignReadVariableOp_3:value:0*
T0*
_output_shapes
: 9
AbsAbsSign:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?H
subSubsub/x:output:0Abs:y:0*
T0*
_output_shapes
: F
add_6AddV2Sign:y:0sub:z:0*
T0*
_output_shapes
: j
Abs_1/ReadVariableOpReadVariableOpreadvariableop_3_resource*
_output_shapes
: *
dtype0O
Abs_1AbsAbs_1/ReadVariableOp:value:0*
T0*
_output_shapes
: M
Less_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_4Less	Abs_1:y:0Less_4/y:output:0*
T0*
_output_shapes
: Q
SelectV2_8/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3i

SelectV2_8SelectV2
Less_4:z:0SelectV2_8/t:output:0	Abs_1:y:0*
T0*
_output_shapes
: M
Less_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_5Less	Abs_1:y:0Less_5/y:output:0*
T0*
_output_shapes
: k
!ones_like_5/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_5Fill*ones_like_5/Shape/shape_as_tensor:output:0ones_like_5/Const:output:0*
T0*
_output_shapes
: L
mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �Y
mul_9Mulones_like_5:output:0mul_9/y:output:0*
T0*
_output_shapes
: F
Log_2LogSelectV2_8:output:0*
T0*
_output_shapes
: P
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_2RealDiv	Log_2:y:0truediv_2/y:output:0*
T0*
_output_shapes
: @
Neg_5Negtruediv_2:z:0*
T0*
_output_shapes
: D
Round_2Roundtruediv_2:z:0*
T0*
_output_shapes
: K
add_7AddV2	Neg_5:y:0Round_2:y:0*
T0*
_output_shapes
: N
StopGradient_3StopGradient	add_7:z:0*
T0*
_output_shapes
: [
add_8AddV2truediv_2:z:0StopGradient_3:output:0*
T0*
_output_shapes
: ^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@v
clip_by_value_2/MinimumMinimum	add_8:z:0"clip_by_value_2/Minimum/y:output:0*
T0*
_output_shapes
: V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*
_output_shapes
: M
mul_10/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
mul_10Mulmul_10/x:output:0clip_by_value_2:z:0*
T0*
_output_shapes
: ^

SelectV2_9SelectV2
Less_5:z:0	mul_9:z:0
mul_10:z:0*
T0*
_output_shapes
: L
pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @X
pow_2Powpow_2/x:output:0SelectV2_9:output:0*
T0*
_output_shapes
: H
mul_11Mul	add_6:z:0	pow_2:z:0*
T0*
_output_shapes
: f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
: *
dtype0K
Neg_6NegReadVariableOp_4:value:0*
T0*
_output_shapes
: J
add_9AddV2	Neg_6:y:0
mul_11:z:0*
T0*
_output_shapes
: M
mul_12/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?P
mul_12Mulmul_12/x:output:0	add_9:z:0*
T0*
_output_shapes
: O
StopGradient_4StopGradient
mul_12:z:0*
T0*
_output_shapes
: f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
: *
dtype0g
add_10AddV2ReadVariableOp_5:value:0StopGradient_4:output:0*
T0*
_output_shapes
: f
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource*
_output_shapes
: *
dtype0M
Sign_1SignReadVariableOp_6:value:0*
T0*
_output_shapes
: =
Abs_2Abs
Sign_1:y:0*
T0*
_output_shapes
: L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_1Subsub_1/x:output:0	Abs_2:y:0*
T0*
_output_shapes
: K
add_11AddV2
Sign_1:y:0	sub_1:z:0*
T0*
_output_shapes
: j
Abs_3/ReadVariableOpReadVariableOpreadvariableop_6_resource*
_output_shapes
: *
dtype0O
Abs_3AbsAbs_3/ReadVariableOp:value:0*
T0*
_output_shapes
: M
Less_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_6Less	Abs_3:y:0Less_6/y:output:0*
T0*
_output_shapes
: R
SelectV2_10/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3k
SelectV2_10SelectV2
Less_6:z:0SelectV2_10/t:output:0	Abs_3:y:0*
T0*
_output_shapes
: M
Less_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_7Less	Abs_3:y:0Less_7/y:output:0*
T0*
_output_shapes
: k
!ones_like_6/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_6Fill*ones_like_6/Shape/shape_as_tensor:output:0ones_like_6/Const:output:0*
T0*
_output_shapes
: M
mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �[
mul_13Mulones_like_6:output:0mul_13/y:output:0*
T0*
_output_shapes
: G
Log_3LogSelectV2_10:output:0*
T0*
_output_shapes
: P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_3RealDiv	Log_3:y:0truediv_3/y:output:0*
T0*
_output_shapes
: @
Neg_7Negtruediv_3:z:0*
T0*
_output_shapes
: D
Round_3Roundtruediv_3:z:0*
T0*
_output_shapes
: L
add_12AddV2	Neg_7:y:0Round_3:y:0*
T0*
_output_shapes
: O
StopGradient_5StopGradient
add_12:z:0*
T0*
_output_shapes
: \
add_13AddV2truediv_3:z:0StopGradient_5:output:0*
T0*
_output_shapes
: ^
clip_by_value_3/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@w
clip_by_value_3/MinimumMinimum
add_13:z:0"clip_by_value_3/Minimum/y:output:0*
T0*
_output_shapes
: V
clip_by_value_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0*
_output_shapes
: M
mul_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
mul_14Mulmul_14/x:output:0clip_by_value_3:z:0*
T0*
_output_shapes
: `
SelectV2_11SelectV2
Less_7:z:0
mul_13:z:0
mul_14:z:0*
T0*
_output_shapes
: L
pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_3Powpow_3/x:output:0SelectV2_11:output:0*
T0*
_output_shapes
: I
mul_15Mul
add_11:z:0	pow_3:z:0*
T0*
_output_shapes
: f
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource*
_output_shapes
: *
dtype0K
Neg_8NegReadVariableOp_7:value:0*
T0*
_output_shapes
: K
add_14AddV2	Neg_8:y:0
mul_15:z:0*
T0*
_output_shapes
: M
mul_16/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
mul_16Mulmul_16/x:output:0
add_14:z:0*
T0*
_output_shapes
: O
StopGradient_6StopGradient
mul_16:z:0*
T0*
_output_shapes
: f
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource*
_output_shapes
: *
dtype0g
add_15AddV2ReadVariableOp_8:value:0StopGradient_6:output:0*
T0*
_output_shapes
: p
Relu_3/ReadVariableOpReadVariableOprelu_3_readvariableop_resource*
_output_shapes
: *
dtype0R
Relu_3ReluRelu_3/ReadVariableOp:value:0*
T0*
_output_shapes
: p
Relu_4/ReadVariableOpReadVariableOprelu_3_readvariableop_resource*
_output_shapes
: *
dtype0R
Relu_4ReluRelu_4/ReadVariableOp:value:0*
T0*
_output_shapes
: M
Less_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3\
Less_8LessRelu_4:activations:0Less_8/y:output:0*
T0*
_output_shapes
: R
SelectV2_12/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3v
SelectV2_12SelectV2
Less_8:z:0SelectV2_12/t:output:0Relu_4:activations:0*
T0*
_output_shapes
: M
Less_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3\
Less_9LessRelu_4:activations:0Less_9/y:output:0*
T0*
_output_shapes
: k
!ones_like_7/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_7Fill*ones_like_7/Shape/shape_as_tensor:output:0ones_like_7/Const:output:0*
T0*
_output_shapes
: M
mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �[
mul_17Mulones_like_7:output:0mul_17/y:output:0*
T0*
_output_shapes
: G
SqrtSqrtSelectV2_12:output:0*
T0*
_output_shapes
: ;
Log_4LogSqrt:y:0*
T0*
_output_shapes
: P
truediv_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_4RealDiv	Log_4:y:0truediv_4/y:output:0*
T0*
_output_shapes
: @
Neg_9Negtruediv_4:z:0*
T0*
_output_shapes
: D
Round_4Roundtruediv_4:z:0*
T0*
_output_shapes
: L
add_16AddV2	Neg_9:y:0Round_4:y:0*
T0*
_output_shapes
: O
StopGradient_7StopGradient
add_16:z:0*
T0*
_output_shapes
: \
add_17AddV2truediv_4:z:0StopGradient_7:output:0*
T0*
_output_shapes
: ^
clip_by_value_4/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Aw
clip_by_value_4/MinimumMinimum
add_17:z:0"clip_by_value_4/Minimum/y:output:0*
T0*
_output_shapes
: V
clip_by_value_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_4Maximumclip_by_value_4/Minimum:z:0clip_by_value_4/y:output:0*
T0*
_output_shapes
: M
mul_18/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Z
mul_18Mulmul_18/x:output:0clip_by_value_4:z:0*
T0*
_output_shapes
: `
SelectV2_13SelectV2
Less_9:z:0
mul_17:z:0
mul_18:z:0*
T0*
_output_shapes
: k
ReadVariableOp_9ReadVariableOprelu_3_readvariableop_resource*
_output_shapes
: *
dtype0L
Neg_10NegReadVariableOp_9:value:0*
T0*
_output_shapes
: ?
Relu_5Relu
Neg_10:y:0*
T0*
_output_shapes
: M
mul_19/yConst*
_output_shapes
: *
dtype0*
valueB
 *    [
mul_19MulRelu_5:activations:0mul_19/y:output:0*
T0*
_output_shapes
: N
	Less_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3T
Less_10Less
mul_19:z:0Less_10/y:output:0*
T0*
_output_shapes
: R
SelectV2_14/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3m
SelectV2_14SelectV2Less_10:z:0SelectV2_14/t:output:0
mul_19:z:0*
T0*
_output_shapes
: N
	Less_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3T
Less_11Less
mul_19:z:0Less_11/y:output:0*
T0*
_output_shapes
: k
!ones_like_8/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_8Fill*ones_like_8/Shape/shape_as_tensor:output:0ones_like_8/Const:output:0*
T0*
_output_shapes
: M
mul_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �[
mul_20Mulones_like_8:output:0mul_20/y:output:0*
T0*
_output_shapes
: I
Sqrt_1SqrtSelectV2_14:output:0*
T0*
_output_shapes
: =
Log_5Log
Sqrt_1:y:0*
T0*
_output_shapes
: P
truediv_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_5RealDiv	Log_5:y:0truediv_5/y:output:0*
T0*
_output_shapes
: A
Neg_11Negtruediv_5:z:0*
T0*
_output_shapes
: D
Round_5Roundtruediv_5:z:0*
T0*
_output_shapes
: M
add_18AddV2
Neg_11:y:0Round_5:y:0*
T0*
_output_shapes
: O
StopGradient_8StopGradient
add_18:z:0*
T0*
_output_shapes
: \
add_19AddV2truediv_5:z:0StopGradient_8:output:0*
T0*
_output_shapes
: ^
clip_by_value_5/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Aw
clip_by_value_5/MinimumMinimum
add_19:z:0"clip_by_value_5/Minimum/y:output:0*
T0*
_output_shapes
: V
clip_by_value_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_5Maximumclip_by_value_5/Minimum:z:0clip_by_value_5/y:output:0*
T0*
_output_shapes
: M
mul_21/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Z
mul_21Mulmul_21/x:output:0clip_by_value_5:z:0*
T0*
_output_shapes
: a
SelectV2_15SelectV2Less_11:z:0
mul_20:z:0
mul_21:z:0*
T0*
_output_shapes
: l
ReadVariableOp_10ReadVariableOprelu_3_readvariableop_resource*
_output_shapes
: *
dtype0U
GreaterEqual_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *    y
GreaterEqual_3GreaterEqualReadVariableOp_10:value:0GreaterEqual_3/y:output:0*
T0*
_output_shapes
: O
LogicalOr_1/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z`
LogicalOr_1	LogicalOrGreaterEqual_3:z:0LogicalOr_1/y:output:0*
_output_shapes
: L
pow_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_4Powpow_4/x:output:0SelectV2_13:output:0*
T0*
_output_shapes
: L
pow_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_5Powpow_5/x:output:0SelectV2_15:output:0*
T0*
_output_shapes
: =
Neg_12Neg	pow_5:z:0*
T0*
_output_shapes
: d
SelectV2_16SelectV2LogicalOr_1:z:0	pow_4:z:0
Neg_12:y:0*
T0*
_output_shapes
: H
Neg_13NegRelu_3:activations:0*
T0*
_output_shapes
: V
add_20AddV2
Neg_13:y:0SelectV2_16:output:0*
T0*
_output_shapes
: M
mul_22/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
mul_22Mulmul_22/x:output:0
add_20:z:0*
T0*
_output_shapes
: O
StopGradient_9StopGradient
mul_22:z:0*
T0*
_output_shapes
: c
add_21AddV2Relu_3:activations:0StopGradient_9:output:0*
T0*
_output_shapes
: T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:a
batchnorm/addAddV2
add_21:z:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: Y
batchnorm/mulMulbatchnorm/Rsqrt:y:0	add_5:z:0*
T0*
_output_shapes
: c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:��������� Z
batchnorm/mul_2Mul
add_15:z:0batchnorm/mul:z:0*
T0*
_output_shapes
: Z
batchnorm/subSub
add_10:z:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^Abs_1/ReadVariableOp^Abs_3/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_10^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8^ReadVariableOp_9^Relu/ReadVariableOp^Relu_1/ReadVariableOp^Relu_3/ReadVariableOp^Relu_4/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 2,
Abs_1/ReadVariableOpAbs_1/ReadVariableOp2,
Abs_3/ReadVariableOpAbs_3/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12&
ReadVariableOp_10ReadVariableOp_102$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82$
ReadVariableOp_9ReadVariableOp_92*
Relu/ReadVariableOpRelu/ReadVariableOp2.
Relu_1/ReadVariableOpRelu_1/ReadVariableOp2.
Relu_3/ReadVariableOpRelu_3/ReadVariableOp2.
Relu_4/ReadVariableOpRelu_4/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�2
�
C__inference_z_mean_layer_call_and_return_conditional_losses_8207036

inputs)
readvariableop_resource:'
readvariableop_3_resource:
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   D[
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes

:N
truedivRealDivmul:z:0Cast:y:0*
T0*
_output_shapes

:@
NegNegtruediv:z:0*
T0*
_output_shapes

:D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:I
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:N
StopGradientStopGradientadd:z:0*
T0*
_output_shapes

:[
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ��Cv
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:R
mul_1MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

:P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   D^
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*
_output_shapes

:L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*
_output_shapes

:h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0O
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes

:M
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*
_output_shapes

:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*
_output_shapes

:R
StopGradient_1StopGradient	mul_3:z:0*
T0*
_output_shapes

:h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0j
add_3AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*
_output_shapes

:U
MatMulMatMulinputs	add_3:z:0*
T0*'
_output_shapes
:���������I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   D]
mul_4MulReadVariableOp_3:value:0mul_4/y:output:0*
T0*
_output_shapes
:P
	truediv_2RealDiv	mul_4:z:0
Cast_1:y:0*
T0*
_output_shapes
:@
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_4AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_4:z:0*
T0*
_output_shapes
:[
add_5AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ��Cv
clip_by_value_1/MinimumMinimum	add_5:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:R
mul_5Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   DZ
	truediv_3RealDiv	mul_5:z:0truediv_3/y:output:0*
T0*
_output_shapes
:L
mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_6Mulmul_6/x:output:0truediv_3:z:0*
T0*
_output_shapes
:f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0K
Neg_3NegReadVariableOp_4:value:0*
T0*
_output_shapes
:I
add_6AddV2	Neg_3:y:0	mul_6:z:0*
T0*
_output_shapes
:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_7Mulmul_7/x:output:0	add_6:z:0*
T0*
_output_shapes
:N
StopGradient_3StopGradient	mul_7:z:0*
T0*
_output_shapes
:f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0f
add_7AddV2ReadVariableOp_5:value:0StopGradient_3:output:0*
T0*
_output_shapes
:a
BiasAddBiasAddMatMul:product:0	add_7:z:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_5:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�	
�
%__inference_BN1_layer_call_fn_8205282

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_BN1_layer_call_and_return_conditional_losses_8204668o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:'#
!
_user_specified_name	8205272:'#
!
_user_specified_name	8205274:'#
!
_user_specified_name	8205276:'#
!
_user_specified_name	8205278
�
C
'__inference_relu2_layer_call_fn_8206890

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_relu2_layer_call_and_return_conditional_losses_8204295`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
4__inference_simplified_encoder_layer_call_fn_8205056

inputs
unknown:, 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_simplified_encoder_layer_call_and_return_conditional_losses_8204990o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������,: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������,
 
_user_specified_nameinputs:'#
!
_user_specified_name	8205026:'#
!
_user_specified_name	8205028:'#
!
_user_specified_name	8205030:'#
!
_user_specified_name	8205032:'#
!
_user_specified_name	8205034:'#
!
_user_specified_name	8205036:'#
!
_user_specified_name	8205038:'#
!
_user_specified_name	8205040:'	#
!
_user_specified_name	8205042:'
#
!
_user_specified_name	8205044:'#
!
_user_specified_name	8205046:'#
!
_user_specified_name	8205048:'#
!
_user_specified_name	8205050:'#
!
_user_specified_name	8205052
��
�
@__inference_BN1_layer_call_and_return_conditional_losses_8205705

inputs%
readvariableop_resource: '
readvariableop_3_resource: '
readvariableop_6_resource: ,
relu_3_readvariableop_resource: 
identity��Abs_1/ReadVariableOp�Abs_3/ReadVariableOp�AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_10�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�ReadVariableOp_6�ReadVariableOp_7�ReadVariableOp_8�ReadVariableOp_9�Relu/ReadVariableOp�Relu_1/ReadVariableOp�Relu_3/ReadVariableOp�Relu_4/ReadVariableOpb
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0P
LessEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ei
	LessEqual	LessEqualReadVariableOp:value:0LessEqual/y:output:0*
T0*
_output_shapes
: g
Relu/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0N
ReluReluRelu/ReadVariableOp:value:0*
T0*
_output_shapes
: l
ones_like/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0i
ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?z
	ones_likeFill(ones_like/Shape/shape_as_tensor:output:0ones_like/Const:output:0*
T0*
_output_shapes
: J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ES
mulMulones_like:output:0mul/y:output:0*
T0*
_output_shapes
: e
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*
_output_shapes
: i
Relu_1/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0R
Relu_1ReluRelu_1/ReadVariableOp:value:0*
T0*
_output_shapes
: K
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3X
LessLessRelu_1:activations:0Less/y:output:0*
T0*
_output_shapes
: Q
SelectV2_1/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3r

SelectV2_1SelectV2Less:z:0SelectV2_1/t:output:0Relu_1:activations:0*
T0*
_output_shapes
: S
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Eo
GreaterEqualGreaterEqualSelectV2_1:output:0GreaterEqual/y:output:0*
T0*
_output_shapes
: k
!ones_like_1/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_1Fill*ones_like_1/Shape/shape_as_tensor:output:0ones_like_1/Const:output:0*
T0*
_output_shapes
: L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   EY
mul_1Mulones_like_1:output:0mul_1/y:output:0*
T0*
_output_shapes
: m

SelectV2_2SelectV2GreaterEqual:z:0	mul_1:z:0SelectV2_1:output:0*
T0*
_output_shapes
: M
Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3\
Less_1LessRelu_1:activations:0Less_1/y:output:0*
T0*
_output_shapes
: k
!ones_like_2/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_2Fill*ones_like_2/Shape/shape_as_tensor:output:0ones_like_2/Const:output:0*
T0*
_output_shapes
: L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �Y
mul_2Mulones_like_2:output:0mul_2/y:output:0*
T0*
_output_shapes
: D
LogLogSelectV2_2:output:0*
T0*
_output_shapes
: N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?T
truedivRealDivLog:y:0truediv/y:output:0*
T0*
_output_shapes
: <
NegNegtruediv:z:0*
T0*
_output_shapes
: @
RoundRoundtruediv:z:0*
T0*
_output_shapes
: E
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes
: J
StopGradientStopGradientadd:z:0*
T0*
_output_shapes
: W
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes
: \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Ar
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
: T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �r
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes
: L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_3Mulmul_3/x:output:0clip_by_value:z:0*
T0*
_output_shapes
: ]

SelectV2_3SelectV2
Less_1:z:0	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: d
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0K
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes
: >
Relu_2Relu	Neg_1:y:0*
T0*
_output_shapes
: L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
mul_4MulRelu_2:activations:0mul_4/y:output:0*
T0*
_output_shapes
: M
Less_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_2Less	mul_4:z:0Less_2/y:output:0*
T0*
_output_shapes
: Q
SelectV2_4/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3i

SelectV2_4SelectV2
Less_2:z:0SelectV2_4/t:output:0	mul_4:z:0*
T0*
_output_shapes
: U
GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Es
GreaterEqual_1GreaterEqualSelectV2_4:output:0GreaterEqual_1/y:output:0*
T0*
_output_shapes
: k
!ones_like_3/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_3Fill*ones_like_3/Shape/shape_as_tensor:output:0ones_like_3/Const:output:0*
T0*
_output_shapes
: L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *   EY
mul_5Mulones_like_3:output:0mul_5/y:output:0*
T0*
_output_shapes
: o

SelectV2_5SelectV2GreaterEqual_1:z:0	mul_5:z:0SelectV2_4:output:0*
T0*
_output_shapes
: M
Less_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_3Less	mul_4:z:0Less_3/y:output:0*
T0*
_output_shapes
: k
!ones_like_4/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_4Fill*ones_like_4/Shape/shape_as_tensor:output:0ones_like_4/Const:output:0*
T0*
_output_shapes
: L
mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �Y
mul_6Mulones_like_4:output:0mul_6/y:output:0*
T0*
_output_shapes
: F
Log_1LogSelectV2_5:output:0*
T0*
_output_shapes
: P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_1RealDiv	Log_1:y:0truediv_1/y:output:0*
T0*
_output_shapes
: @
Neg_2Negtruediv_1:z:0*
T0*
_output_shapes
: D
Round_1Roundtruediv_1:z:0*
T0*
_output_shapes
: K
add_2AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
: N
StopGradient_1StopGradient	add_2:z:0*
T0*
_output_shapes
: [
add_3AddV2truediv_1:z:0StopGradient_1:output:0*
T0*
_output_shapes
: ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Av
clip_by_value_1/MinimumMinimum	add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
: V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
: L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?X
mul_7Mulmul_7/x:output:0clip_by_value_1:z:0*
T0*
_output_shapes
: ]

SelectV2_6SelectV2
Less_3:z:0	mul_6:z:0	mul_7:z:0*
T0*
_output_shapes
: d
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0U
GreaterEqual_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    x
GreaterEqual_2GreaterEqualReadVariableOp_2:value:0GreaterEqual_2/y:output:0*
T0*
_output_shapes
: M
LogicalOr/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z\
	LogicalOr	LogicalOrGreaterEqual_2:z:0LogicalOr/y:output:0*
_output_shapes
: J
pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
powPowpow/x:output:0SelectV2_3:output:0*
T0*
_output_shapes
: L
pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @X
pow_1Powpow_1/x:output:0SelectV2_6:output:0*
T0*
_output_shapes
: <
Neg_3Neg	pow_1:z:0*
T0*
_output_shapes
: ^

SelectV2_7SelectV2LogicalOr:z:0pow:z:0	Neg_3:y:0*
T0*
_output_shapes
: D
Neg_4NegSelectV2:output:0*
T0*
_output_shapes
: S
add_4AddV2	Neg_4:y:0SelectV2_7:output:0*
T0*
_output_shapes
: L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_8Mulmul_8/x:output:0	add_4:z:0*
T0*
_output_shapes
: N
StopGradient_2StopGradient	mul_8:z:0*
T0*
_output_shapes
: _
add_5AddV2SelectV2:output:0StopGradient_2:output:0*
T0*
_output_shapes
: f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
: *
dtype0K
SignSignReadVariableOp_3:value:0*
T0*
_output_shapes
: 9
AbsAbsSign:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?H
subSubsub/x:output:0Abs:y:0*
T0*
_output_shapes
: F
add_6AddV2Sign:y:0sub:z:0*
T0*
_output_shapes
: j
Abs_1/ReadVariableOpReadVariableOpreadvariableop_3_resource*
_output_shapes
: *
dtype0O
Abs_1AbsAbs_1/ReadVariableOp:value:0*
T0*
_output_shapes
: M
Less_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_4Less	Abs_1:y:0Less_4/y:output:0*
T0*
_output_shapes
: Q
SelectV2_8/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3i

SelectV2_8SelectV2
Less_4:z:0SelectV2_8/t:output:0	Abs_1:y:0*
T0*
_output_shapes
: M
Less_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_5Less	Abs_1:y:0Less_5/y:output:0*
T0*
_output_shapes
: k
!ones_like_5/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_5Fill*ones_like_5/Shape/shape_as_tensor:output:0ones_like_5/Const:output:0*
T0*
_output_shapes
: L
mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �Y
mul_9Mulones_like_5:output:0mul_9/y:output:0*
T0*
_output_shapes
: F
Log_2LogSelectV2_8:output:0*
T0*
_output_shapes
: P
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_2RealDiv	Log_2:y:0truediv_2/y:output:0*
T0*
_output_shapes
: @
Neg_5Negtruediv_2:z:0*
T0*
_output_shapes
: D
Round_2Roundtruediv_2:z:0*
T0*
_output_shapes
: K
add_7AddV2	Neg_5:y:0Round_2:y:0*
T0*
_output_shapes
: N
StopGradient_3StopGradient	add_7:z:0*
T0*
_output_shapes
: [
add_8AddV2truediv_2:z:0StopGradient_3:output:0*
T0*
_output_shapes
: ^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@v
clip_by_value_2/MinimumMinimum	add_8:z:0"clip_by_value_2/Minimum/y:output:0*
T0*
_output_shapes
: V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*
_output_shapes
: M
mul_10/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
mul_10Mulmul_10/x:output:0clip_by_value_2:z:0*
T0*
_output_shapes
: ^

SelectV2_9SelectV2
Less_5:z:0	mul_9:z:0
mul_10:z:0*
T0*
_output_shapes
: L
pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @X
pow_2Powpow_2/x:output:0SelectV2_9:output:0*
T0*
_output_shapes
: H
mul_11Mul	add_6:z:0	pow_2:z:0*
T0*
_output_shapes
: f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
: *
dtype0K
Neg_6NegReadVariableOp_4:value:0*
T0*
_output_shapes
: J
add_9AddV2	Neg_6:y:0
mul_11:z:0*
T0*
_output_shapes
: M
mul_12/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?P
mul_12Mulmul_12/x:output:0	add_9:z:0*
T0*
_output_shapes
: O
StopGradient_4StopGradient
mul_12:z:0*
T0*
_output_shapes
: f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
: *
dtype0g
add_10AddV2ReadVariableOp_5:value:0StopGradient_4:output:0*
T0*
_output_shapes
: f
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource*
_output_shapes
: *
dtype0M
Sign_1SignReadVariableOp_6:value:0*
T0*
_output_shapes
: =
Abs_2Abs
Sign_1:y:0*
T0*
_output_shapes
: L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_1Subsub_1/x:output:0	Abs_2:y:0*
T0*
_output_shapes
: K
add_11AddV2
Sign_1:y:0	sub_1:z:0*
T0*
_output_shapes
: j
Abs_3/ReadVariableOpReadVariableOpreadvariableop_6_resource*
_output_shapes
: *
dtype0O
Abs_3AbsAbs_3/ReadVariableOp:value:0*
T0*
_output_shapes
: M
Less_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_6Less	Abs_3:y:0Less_6/y:output:0*
T0*
_output_shapes
: R
SelectV2_10/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3k
SelectV2_10SelectV2
Less_6:z:0SelectV2_10/t:output:0	Abs_3:y:0*
T0*
_output_shapes
: M
Less_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_7Less	Abs_3:y:0Less_7/y:output:0*
T0*
_output_shapes
: k
!ones_like_6/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_6Fill*ones_like_6/Shape/shape_as_tensor:output:0ones_like_6/Const:output:0*
T0*
_output_shapes
: M
mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �[
mul_13Mulones_like_6:output:0mul_13/y:output:0*
T0*
_output_shapes
: G
Log_3LogSelectV2_10:output:0*
T0*
_output_shapes
: P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_3RealDiv	Log_3:y:0truediv_3/y:output:0*
T0*
_output_shapes
: @
Neg_7Negtruediv_3:z:0*
T0*
_output_shapes
: D
Round_3Roundtruediv_3:z:0*
T0*
_output_shapes
: L
add_12AddV2	Neg_7:y:0Round_3:y:0*
T0*
_output_shapes
: O
StopGradient_5StopGradient
add_12:z:0*
T0*
_output_shapes
: \
add_13AddV2truediv_3:z:0StopGradient_5:output:0*
T0*
_output_shapes
: ^
clip_by_value_3/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@w
clip_by_value_3/MinimumMinimum
add_13:z:0"clip_by_value_3/Minimum/y:output:0*
T0*
_output_shapes
: V
clip_by_value_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0*
_output_shapes
: M
mul_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
mul_14Mulmul_14/x:output:0clip_by_value_3:z:0*
T0*
_output_shapes
: `
SelectV2_11SelectV2
Less_7:z:0
mul_13:z:0
mul_14:z:0*
T0*
_output_shapes
: L
pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_3Powpow_3/x:output:0SelectV2_11:output:0*
T0*
_output_shapes
: I
mul_15Mul
add_11:z:0	pow_3:z:0*
T0*
_output_shapes
: f
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource*
_output_shapes
: *
dtype0K
Neg_8NegReadVariableOp_7:value:0*
T0*
_output_shapes
: K
add_14AddV2	Neg_8:y:0
mul_15:z:0*
T0*
_output_shapes
: M
mul_16/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
mul_16Mulmul_16/x:output:0
add_14:z:0*
T0*
_output_shapes
: O
StopGradient_6StopGradient
mul_16:z:0*
T0*
_output_shapes
: f
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource*
_output_shapes
: *
dtype0g
add_15AddV2ReadVariableOp_8:value:0StopGradient_6:output:0*
T0*
_output_shapes
: p
Relu_3/ReadVariableOpReadVariableOprelu_3_readvariableop_resource*
_output_shapes
: *
dtype0R
Relu_3ReluRelu_3/ReadVariableOp:value:0*
T0*
_output_shapes
: p
Relu_4/ReadVariableOpReadVariableOprelu_3_readvariableop_resource*
_output_shapes
: *
dtype0R
Relu_4ReluRelu_4/ReadVariableOp:value:0*
T0*
_output_shapes
: M
Less_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3\
Less_8LessRelu_4:activations:0Less_8/y:output:0*
T0*
_output_shapes
: R
SelectV2_12/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3v
SelectV2_12SelectV2
Less_8:z:0SelectV2_12/t:output:0Relu_4:activations:0*
T0*
_output_shapes
: M
Less_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3\
Less_9LessRelu_4:activations:0Less_9/y:output:0*
T0*
_output_shapes
: k
!ones_like_7/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_7Fill*ones_like_7/Shape/shape_as_tensor:output:0ones_like_7/Const:output:0*
T0*
_output_shapes
: M
mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �[
mul_17Mulones_like_7:output:0mul_17/y:output:0*
T0*
_output_shapes
: G
SqrtSqrtSelectV2_12:output:0*
T0*
_output_shapes
: ;
Log_4LogSqrt:y:0*
T0*
_output_shapes
: P
truediv_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_4RealDiv	Log_4:y:0truediv_4/y:output:0*
T0*
_output_shapes
: @
Neg_9Negtruediv_4:z:0*
T0*
_output_shapes
: D
Round_4Roundtruediv_4:z:0*
T0*
_output_shapes
: L
add_16AddV2	Neg_9:y:0Round_4:y:0*
T0*
_output_shapes
: O
StopGradient_7StopGradient
add_16:z:0*
T0*
_output_shapes
: \
add_17AddV2truediv_4:z:0StopGradient_7:output:0*
T0*
_output_shapes
: ^
clip_by_value_4/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Aw
clip_by_value_4/MinimumMinimum
add_17:z:0"clip_by_value_4/Minimum/y:output:0*
T0*
_output_shapes
: V
clip_by_value_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_4Maximumclip_by_value_4/Minimum:z:0clip_by_value_4/y:output:0*
T0*
_output_shapes
: M
mul_18/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Z
mul_18Mulmul_18/x:output:0clip_by_value_4:z:0*
T0*
_output_shapes
: `
SelectV2_13SelectV2
Less_9:z:0
mul_17:z:0
mul_18:z:0*
T0*
_output_shapes
: k
ReadVariableOp_9ReadVariableOprelu_3_readvariableop_resource*
_output_shapes
: *
dtype0L
Neg_10NegReadVariableOp_9:value:0*
T0*
_output_shapes
: ?
Relu_5Relu
Neg_10:y:0*
T0*
_output_shapes
: M
mul_19/yConst*
_output_shapes
: *
dtype0*
valueB
 *    [
mul_19MulRelu_5:activations:0mul_19/y:output:0*
T0*
_output_shapes
: N
	Less_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3T
Less_10Less
mul_19:z:0Less_10/y:output:0*
T0*
_output_shapes
: R
SelectV2_14/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3m
SelectV2_14SelectV2Less_10:z:0SelectV2_14/t:output:0
mul_19:z:0*
T0*
_output_shapes
: N
	Less_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3T
Less_11Less
mul_19:z:0Less_11/y:output:0*
T0*
_output_shapes
: k
!ones_like_8/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_8Fill*ones_like_8/Shape/shape_as_tensor:output:0ones_like_8/Const:output:0*
T0*
_output_shapes
: M
mul_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �[
mul_20Mulones_like_8:output:0mul_20/y:output:0*
T0*
_output_shapes
: I
Sqrt_1SqrtSelectV2_14:output:0*
T0*
_output_shapes
: =
Log_5Log
Sqrt_1:y:0*
T0*
_output_shapes
: P
truediv_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_5RealDiv	Log_5:y:0truediv_5/y:output:0*
T0*
_output_shapes
: A
Neg_11Negtruediv_5:z:0*
T0*
_output_shapes
: D
Round_5Roundtruediv_5:z:0*
T0*
_output_shapes
: M
add_18AddV2
Neg_11:y:0Round_5:y:0*
T0*
_output_shapes
: O
StopGradient_8StopGradient
add_18:z:0*
T0*
_output_shapes
: \
add_19AddV2truediv_5:z:0StopGradient_8:output:0*
T0*
_output_shapes
: ^
clip_by_value_5/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Aw
clip_by_value_5/MinimumMinimum
add_19:z:0"clip_by_value_5/Minimum/y:output:0*
T0*
_output_shapes
: V
clip_by_value_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_5Maximumclip_by_value_5/Minimum:z:0clip_by_value_5/y:output:0*
T0*
_output_shapes
: M
mul_21/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Z
mul_21Mulmul_21/x:output:0clip_by_value_5:z:0*
T0*
_output_shapes
: a
SelectV2_15SelectV2Less_11:z:0
mul_20:z:0
mul_21:z:0*
T0*
_output_shapes
: l
ReadVariableOp_10ReadVariableOprelu_3_readvariableop_resource*
_output_shapes
: *
dtype0U
GreaterEqual_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *    y
GreaterEqual_3GreaterEqualReadVariableOp_10:value:0GreaterEqual_3/y:output:0*
T0*
_output_shapes
: O
LogicalOr_1/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z`
LogicalOr_1	LogicalOrGreaterEqual_3:z:0LogicalOr_1/y:output:0*
_output_shapes
: L
pow_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_4Powpow_4/x:output:0SelectV2_13:output:0*
T0*
_output_shapes
: L
pow_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_5Powpow_5/x:output:0SelectV2_15:output:0*
T0*
_output_shapes
: =
Neg_12Neg	pow_5:z:0*
T0*
_output_shapes
: d
SelectV2_16SelectV2LogicalOr_1:z:0	pow_4:z:0
Neg_12:y:0*
T0*
_output_shapes
: H
Neg_13NegRelu_3:activations:0*
T0*
_output_shapes
: V
add_20AddV2
Neg_13:y:0SelectV2_16:output:0*
T0*
_output_shapes
: M
mul_22/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
mul_22Mulmul_22/x:output:0
add_20:z:0*
T0*
_output_shapes
: O
StopGradient_9StopGradient
mul_22:z:0*
T0*
_output_shapes
: c
add_21AddV2Relu_3:activations:0StopGradient_9:output:0*
T0*
_output_shapes
: h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

: �
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:��������� l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 M
Sign_2Signmoments/Squeeze:output:0*
T0*
_output_shapes
: =
Abs_4Abs
Sign_2:y:0*
T0*
_output_shapes
: L
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_2Subsub_2/x:output:0	Abs_4:y:0*
T0*
_output_shapes
: K
add_22AddV2
Sign_2:y:0	sub_2:z:0*
T0*
_output_shapes
: K
Abs_5Absmoments/Squeeze:output:0*
T0*
_output_shapes
: N
	Less_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3S
Less_12Less	Abs_5:y:0Less_12/y:output:0*
T0*
_output_shapes
: R
SelectV2_17/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3l
SelectV2_17SelectV2Less_12:z:0SelectV2_17/t:output:0	Abs_5:y:0*
T0*
_output_shapes
: N
	Less_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3S
Less_13Less	Abs_5:y:0Less_13/y:output:0*
T0*
_output_shapes
: k
!ones_like_9/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: V
ones_like_9/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_9Fill*ones_like_9/Shape/shape_as_tensor:output:0ones_like_9/Const:output:0*
T0*
_output_shapes
: M
mul_23/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �[
mul_23Mulones_like_9:output:0mul_23/y:output:0*
T0*
_output_shapes
: G
Log_6LogSelectV2_17:output:0*
T0*
_output_shapes
: P
truediv_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_6RealDiv	Log_6:y:0truediv_6/y:output:0*
T0*
_output_shapes
: A
Neg_14Negtruediv_6:z:0*
T0*
_output_shapes
: D
Round_6Roundtruediv_6:z:0*
T0*
_output_shapes
: M
add_23AddV2
Neg_14:y:0Round_6:y:0*
T0*
_output_shapes
: P
StopGradient_10StopGradient
add_23:z:0*
T0*
_output_shapes
: ]
add_24AddV2truediv_6:z:0StopGradient_10:output:0*
T0*
_output_shapes
: ^
clip_by_value_6/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@w
clip_by_value_6/MinimumMinimum
add_24:z:0"clip_by_value_6/Minimum/y:output:0*
T0*
_output_shapes
: V
clip_by_value_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_6Maximumclip_by_value_6/Minimum:z:0clip_by_value_6/y:output:0*
T0*
_output_shapes
: M
mul_24/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
mul_24Mulmul_24/x:output:0clip_by_value_6:z:0*
T0*
_output_shapes
: a
SelectV2_18SelectV2Less_13:z:0
mul_23:z:0
mul_24:z:0*
T0*
_output_shapes
: L
pow_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_6Powpow_6/x:output:0SelectV2_18:output:0*
T0*
_output_shapes
: I
mul_25Mul
add_22:z:0	pow_6:z:0*
T0*
_output_shapes
: L
Neg_15Negmoments/Squeeze:output:0*
T0*
_output_shapes
: L
add_25AddV2
Neg_15:y:0
mul_25:z:0*
T0*
_output_shapes
: M
mul_26/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
mul_26Mulmul_26/x:output:0
add_25:z:0*
T0*
_output_shapes
: P
StopGradient_11StopGradient
mul_26:z:0*
T0*
_output_shapes
: h
add_26AddV2moments/Squeeze:output:0StopGradient_11:output:0*
T0*
_output_shapes
: O
Relu_6Relumoments/Squeeze_1:output:0*
T0*
_output_shapes
: O
Relu_7Relumoments/Squeeze_1:output:0*
T0*
_output_shapes
: N
	Less_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3^
Less_14LessRelu_7:activations:0Less_14/y:output:0*
T0*
_output_shapes
: R
SelectV2_19/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3w
SelectV2_19SelectV2Less_14:z:0SelectV2_19/t:output:0Relu_7:activations:0*
T0*
_output_shapes
: N
	Less_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3^
Less_15LessRelu_7:activations:0Less_15/y:output:0*
T0*
_output_shapes
: l
"ones_like_10/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: W
ones_like_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_10Fill+ones_like_10/Shape/shape_as_tensor:output:0ones_like_10/Const:output:0*
T0*
_output_shapes
: M
mul_27/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �\
mul_27Mulones_like_10:output:0mul_27/y:output:0*
T0*
_output_shapes
: I
Sqrt_2SqrtSelectV2_19:output:0*
T0*
_output_shapes
: =
Log_7Log
Sqrt_2:y:0*
T0*
_output_shapes
: P
truediv_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_7RealDiv	Log_7:y:0truediv_7/y:output:0*
T0*
_output_shapes
: A
Neg_16Negtruediv_7:z:0*
T0*
_output_shapes
: D
Round_7Roundtruediv_7:z:0*
T0*
_output_shapes
: M
add_27AddV2
Neg_16:y:0Round_7:y:0*
T0*
_output_shapes
: P
StopGradient_12StopGradient
add_27:z:0*
T0*
_output_shapes
: ]
add_28AddV2truediv_7:z:0StopGradient_12:output:0*
T0*
_output_shapes
: ^
clip_by_value_7/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Aw
clip_by_value_7/MinimumMinimum
add_28:z:0"clip_by_value_7/Minimum/y:output:0*
T0*
_output_shapes
: V
clip_by_value_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_7Maximumclip_by_value_7/Minimum:z:0clip_by_value_7/y:output:0*
T0*
_output_shapes
: M
mul_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Z
mul_28Mulmul_28/x:output:0clip_by_value_7:z:0*
T0*
_output_shapes
: a
SelectV2_20SelectV2Less_15:z:0
mul_27:z:0
mul_28:z:0*
T0*
_output_shapes
: N
Neg_17Negmoments/Squeeze_1:output:0*
T0*
_output_shapes
: ?
Relu_8Relu
Neg_17:y:0*
T0*
_output_shapes
: M
mul_29/yConst*
_output_shapes
: *
dtype0*
valueB
 *    [
mul_29MulRelu_8:activations:0mul_29/y:output:0*
T0*
_output_shapes
: N
	Less_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3T
Less_16Less
mul_29:z:0Less_16/y:output:0*
T0*
_output_shapes
: R
SelectV2_21/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3m
SelectV2_21SelectV2Less_16:z:0SelectV2_21/t:output:0
mul_29:z:0*
T0*
_output_shapes
: N
	Less_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3T
Less_17Less
mul_29:z:0Less_17/y:output:0*
T0*
_output_shapes
: l
"ones_like_11/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: W
ones_like_11/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_11Fill+ones_like_11/Shape/shape_as_tensor:output:0ones_like_11/Const:output:0*
T0*
_output_shapes
: M
mul_30/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �\
mul_30Mulones_like_11:output:0mul_30/y:output:0*
T0*
_output_shapes
: I
Sqrt_3SqrtSelectV2_21:output:0*
T0*
_output_shapes
: =
Log_8Log
Sqrt_3:y:0*
T0*
_output_shapes
: P
truediv_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_8RealDiv	Log_8:y:0truediv_8/y:output:0*
T0*
_output_shapes
: A
Neg_18Negtruediv_8:z:0*
T0*
_output_shapes
: D
Round_8Roundtruediv_8:z:0*
T0*
_output_shapes
: M
add_29AddV2
Neg_18:y:0Round_8:y:0*
T0*
_output_shapes
: P
StopGradient_13StopGradient
add_29:z:0*
T0*
_output_shapes
: ]
add_30AddV2truediv_8:z:0StopGradient_13:output:0*
T0*
_output_shapes
: ^
clip_by_value_8/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Aw
clip_by_value_8/MinimumMinimum
add_30:z:0"clip_by_value_8/Minimum/y:output:0*
T0*
_output_shapes
: V
clip_by_value_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_8Maximumclip_by_value_8/Minimum:z:0clip_by_value_8/y:output:0*
T0*
_output_shapes
: M
mul_31/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Z
mul_31Mulmul_31/x:output:0clip_by_value_8:z:0*
T0*
_output_shapes
: a
SelectV2_22SelectV2Less_17:z:0
mul_30:z:0
mul_31:z:0*
T0*
_output_shapes
: U
GreaterEqual_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *    z
GreaterEqual_4GreaterEqualmoments/Squeeze_1:output:0GreaterEqual_4/y:output:0*
T0*
_output_shapes
: O
LogicalOr_2/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z`
LogicalOr_2	LogicalOrGreaterEqual_4:z:0LogicalOr_2/y:output:0*
_output_shapes
: L
pow_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_7Powpow_7/x:output:0SelectV2_20:output:0*
T0*
_output_shapes
: L
pow_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_8Powpow_8/x:output:0SelectV2_22:output:0*
T0*
_output_shapes
: =
Neg_19Neg	pow_8:z:0*
T0*
_output_shapes
: d
SelectV2_23SelectV2LogicalOr_2:z:0	pow_7:z:0
Neg_19:y:0*
T0*
_output_shapes
: H
Neg_20NegRelu_6:activations:0*
T0*
_output_shapes
: V
add_31AddV2
Neg_20:y:0SelectV2_23:output:0*
T0*
_output_shapes
: M
mul_32/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
mul_32Mulmul_32/x:output:0
add_31:z:0*
T0*
_output_shapes
: P
StopGradient_14StopGradient
mul_32:z:0*
T0*
_output_shapes
: d
add_32AddV2Relu_6:activations:0StopGradient_14:output:0*
T0*
_output_shapes
: Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<t
AssignMovingAvg/ReadVariableOpReadVariableOpreadvariableop_6_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvgAssignSubVariableOpreadvariableop_6_resourceAssignMovingAvg/mul:z:0^Abs_3/ReadVariableOp^AssignMovingAvg/ReadVariableOp^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<{
 AssignMovingAvg_1/ReadVariableOpReadVariableOprelu_3_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: ~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvg_1AssignSubVariableOprelu_3_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp_10^ReadVariableOp_9^Relu_3/ReadVariableOp^Relu_4/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:a
batchnorm/addAddV2
add_32:z:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: Y
batchnorm/mulMulbatchnorm/Rsqrt:y:0	add_5:z:0*
T0*
_output_shapes
: c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:��������� Z
batchnorm/mul_2Mul
add_26:z:0batchnorm/mul:z:0*
T0*
_output_shapes
: Z
batchnorm/subSub
add_10:z:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^Abs_1/ReadVariableOp^Abs_3/ReadVariableOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_10^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8^ReadVariableOp_9^Relu/ReadVariableOp^Relu_1/ReadVariableOp^Relu_3/ReadVariableOp^Relu_4/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 2,
Abs_1/ReadVariableOpAbs_1/ReadVariableOp2,
Abs_3/ReadVariableOpAbs_3/ReadVariableOp2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12&
ReadVariableOp_10ReadVariableOp_102$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82$
ReadVariableOp_9ReadVariableOp_92*
Relu/ReadVariableOpRelu/ReadVariableOp2.
Relu_1/ReadVariableOpRelu_1/ReadVariableOp2.
Relu_3/ReadVariableOpRelu_3/ReadVariableOp2.
Relu_4/ReadVariableOpRelu_4/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�	
�
%__inference_BN1_layer_call_fn_8205269

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_BN1_layer_call_and_return_conditional_losses_8203642o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:'#
!
_user_specified_name	8205259:'#
!
_user_specified_name	8205261:'#
!
_user_specified_name	8205263:'#
!
_user_specified_name	8205265
�
C
'__inference_relu1_layer_call_fn_8206000

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_relu1_layer_call_and_return_conditional_losses_8203720`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
��
�
@__inference_BN2_layer_call_and_return_conditional_losses_8204973

inputs%
readvariableop_resource:'
readvariableop_3_resource:'
readvariableop_6_resource:,
relu_3_readvariableop_resource:
identity��Abs_1/ReadVariableOp�Abs_3/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_10�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5�ReadVariableOp_6�ReadVariableOp_7�ReadVariableOp_8�ReadVariableOp_9�Relu/ReadVariableOp�Relu_1/ReadVariableOp�Relu_3/ReadVariableOp�Relu_4/ReadVariableOpb
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0P
LessEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ei
	LessEqual	LessEqualReadVariableOp:value:0LessEqual/y:output:0*
T0*
_output_shapes
:g
Relu/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0N
ReluReluRelu/ReadVariableOp:value:0*
T0*
_output_shapes
:l
ones_like/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0i
ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?z
	ones_likeFill(ones_like/Shape/shape_as_tensor:output:0ones_like/Const:output:0*
T0*
_output_shapes
:J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ES
mulMulones_like:output:0mul/y:output:0*
T0*
_output_shapes
:e
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*
_output_shapes
:i
Relu_1/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0R
Relu_1ReluRelu_1/ReadVariableOp:value:0*
T0*
_output_shapes
:K
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3X
LessLessRelu_1:activations:0Less/y:output:0*
T0*
_output_shapes
:Q
SelectV2_1/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3r

SelectV2_1SelectV2Less:z:0SelectV2_1/t:output:0Relu_1:activations:0*
T0*
_output_shapes
:S
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Eo
GreaterEqualGreaterEqualSelectV2_1:output:0GreaterEqual/y:output:0*
T0*
_output_shapes
:k
!ones_like_1/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_1Fill*ones_like_1/Shape/shape_as_tensor:output:0ones_like_1/Const:output:0*
T0*
_output_shapes
:L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   EY
mul_1Mulones_like_1:output:0mul_1/y:output:0*
T0*
_output_shapes
:m

SelectV2_2SelectV2GreaterEqual:z:0	mul_1:z:0SelectV2_1:output:0*
T0*
_output_shapes
:M
Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3\
Less_1LessRelu_1:activations:0Less_1/y:output:0*
T0*
_output_shapes
:k
!ones_like_2/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_2Fill*ones_like_2/Shape/shape_as_tensor:output:0ones_like_2/Const:output:0*
T0*
_output_shapes
:L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �Y
mul_2Mulones_like_2:output:0mul_2/y:output:0*
T0*
_output_shapes
:D
LogLogSelectV2_2:output:0*
T0*
_output_shapes
:N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?T
truedivRealDivLog:y:0truediv/y:output:0*
T0*
_output_shapes
:<
NegNegtruediv:z:0*
T0*
_output_shapes
:@
RoundRoundtruediv:z:0*
T0*
_output_shapes
:E
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes
:J
StopGradientStopGradientadd:z:0*
T0*
_output_shapes
:W
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes
:\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Ar
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �r
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes
:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_3Mulmul_3/x:output:0clip_by_value:z:0*
T0*
_output_shapes
:]

SelectV2_3SelectV2
Less_1:z:0	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:d
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0K
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes
:>
Relu_2Relu	Neg_1:y:0*
T0*
_output_shapes
:L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
mul_4MulRelu_2:activations:0mul_4/y:output:0*
T0*
_output_shapes
:M
Less_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_2Less	mul_4:z:0Less_2/y:output:0*
T0*
_output_shapes
:Q
SelectV2_4/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3i

SelectV2_4SelectV2
Less_2:z:0SelectV2_4/t:output:0	mul_4:z:0*
T0*
_output_shapes
:U
GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Es
GreaterEqual_1GreaterEqualSelectV2_4:output:0GreaterEqual_1/y:output:0*
T0*
_output_shapes
:k
!ones_like_3/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_3Fill*ones_like_3/Shape/shape_as_tensor:output:0ones_like_3/Const:output:0*
T0*
_output_shapes
:L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *   EY
mul_5Mulones_like_3:output:0mul_5/y:output:0*
T0*
_output_shapes
:o

SelectV2_5SelectV2GreaterEqual_1:z:0	mul_5:z:0SelectV2_4:output:0*
T0*
_output_shapes
:M
Less_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_3Less	mul_4:z:0Less_3/y:output:0*
T0*
_output_shapes
:k
!ones_like_4/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_4Fill*ones_like_4/Shape/shape_as_tensor:output:0ones_like_4/Const:output:0*
T0*
_output_shapes
:L
mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �Y
mul_6Mulones_like_4:output:0mul_6/y:output:0*
T0*
_output_shapes
:F
Log_1LogSelectV2_5:output:0*
T0*
_output_shapes
:P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_1RealDiv	Log_1:y:0truediv_1/y:output:0*
T0*
_output_shapes
:@
Neg_2Negtruediv_1:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_1:z:0*
T0*
_output_shapes
:K
add_2AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_1StopGradient	add_2:z:0*
T0*
_output_shapes
:[
add_3AddV2truediv_1:z:0StopGradient_1:output:0*
T0*
_output_shapes
:^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Av
clip_by_value_1/MinimumMinimum	add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?X
mul_7Mulmul_7/x:output:0clip_by_value_1:z:0*
T0*
_output_shapes
:]

SelectV2_6SelectV2
Less_3:z:0	mul_6:z:0	mul_7:z:0*
T0*
_output_shapes
:d
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0U
GreaterEqual_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    x
GreaterEqual_2GreaterEqualReadVariableOp_2:value:0GreaterEqual_2/y:output:0*
T0*
_output_shapes
:M
LogicalOr/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z\
	LogicalOr	LogicalOrGreaterEqual_2:z:0LogicalOr/y:output:0*
_output_shapes
:J
pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
powPowpow/x:output:0SelectV2_3:output:0*
T0*
_output_shapes
:L
pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @X
pow_1Powpow_1/x:output:0SelectV2_6:output:0*
T0*
_output_shapes
:<
Neg_3Neg	pow_1:z:0*
T0*
_output_shapes
:^

SelectV2_7SelectV2LogicalOr:z:0pow:z:0	Neg_3:y:0*
T0*
_output_shapes
:D
Neg_4NegSelectV2:output:0*
T0*
_output_shapes
:S
add_4AddV2	Neg_4:y:0SelectV2_7:output:0*
T0*
_output_shapes
:L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_8Mulmul_8/x:output:0	add_4:z:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	mul_8:z:0*
T0*
_output_shapes
:_
add_5AddV2SelectV2:output:0StopGradient_2:output:0*
T0*
_output_shapes
:f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0K
SignSignReadVariableOp_3:value:0*
T0*
_output_shapes
:9
AbsAbsSign:y:0*
T0*
_output_shapes
:J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?H
subSubsub/x:output:0Abs:y:0*
T0*
_output_shapes
:F
add_6AddV2Sign:y:0sub:z:0*
T0*
_output_shapes
:j
Abs_1/ReadVariableOpReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0O
Abs_1AbsAbs_1/ReadVariableOp:value:0*
T0*
_output_shapes
:M
Less_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_4Less	Abs_1:y:0Less_4/y:output:0*
T0*
_output_shapes
:Q
SelectV2_8/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3i

SelectV2_8SelectV2
Less_4:z:0SelectV2_8/t:output:0	Abs_1:y:0*
T0*
_output_shapes
:M
Less_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_5Less	Abs_1:y:0Less_5/y:output:0*
T0*
_output_shapes
:k
!ones_like_5/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_5Fill*ones_like_5/Shape/shape_as_tensor:output:0ones_like_5/Const:output:0*
T0*
_output_shapes
:L
mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �Y
mul_9Mulones_like_5:output:0mul_9/y:output:0*
T0*
_output_shapes
:F
Log_2LogSelectV2_8:output:0*
T0*
_output_shapes
:P
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_2RealDiv	Log_2:y:0truediv_2/y:output:0*
T0*
_output_shapes
:@
Neg_5Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_2Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_7AddV2	Neg_5:y:0Round_2:y:0*
T0*
_output_shapes
:N
StopGradient_3StopGradient	add_7:z:0*
T0*
_output_shapes
:[
add_8AddV2truediv_2:z:0StopGradient_3:output:0*
T0*
_output_shapes
:^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@v
clip_by_value_2/MinimumMinimum	add_8:z:0"clip_by_value_2/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*
_output_shapes
:M
mul_10/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
mul_10Mulmul_10/x:output:0clip_by_value_2:z:0*
T0*
_output_shapes
:^

SelectV2_9SelectV2
Less_5:z:0	mul_9:z:0
mul_10:z:0*
T0*
_output_shapes
:L
pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @X
pow_2Powpow_2/x:output:0SelectV2_9:output:0*
T0*
_output_shapes
:H
mul_11Mul	add_6:z:0	pow_2:z:0*
T0*
_output_shapes
:f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0K
Neg_6NegReadVariableOp_4:value:0*
T0*
_output_shapes
:J
add_9AddV2	Neg_6:y:0
mul_11:z:0*
T0*
_output_shapes
:M
mul_12/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?P
mul_12Mulmul_12/x:output:0	add_9:z:0*
T0*
_output_shapes
:O
StopGradient_4StopGradient
mul_12:z:0*
T0*
_output_shapes
:f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0g
add_10AddV2ReadVariableOp_5:value:0StopGradient_4:output:0*
T0*
_output_shapes
:f
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource*
_output_shapes
:*
dtype0M
Sign_1SignReadVariableOp_6:value:0*
T0*
_output_shapes
:=
Abs_2Abs
Sign_1:y:0*
T0*
_output_shapes
:L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_1Subsub_1/x:output:0	Abs_2:y:0*
T0*
_output_shapes
:K
add_11AddV2
Sign_1:y:0	sub_1:z:0*
T0*
_output_shapes
:j
Abs_3/ReadVariableOpReadVariableOpreadvariableop_6_resource*
_output_shapes
:*
dtype0O
Abs_3AbsAbs_3/ReadVariableOp:value:0*
T0*
_output_shapes
:M
Less_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_6Less	Abs_3:y:0Less_6/y:output:0*
T0*
_output_shapes
:R
SelectV2_10/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3k
SelectV2_10SelectV2
Less_6:z:0SelectV2_10/t:output:0	Abs_3:y:0*
T0*
_output_shapes
:M
Less_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3Q
Less_7Less	Abs_3:y:0Less_7/y:output:0*
T0*
_output_shapes
:k
!ones_like_6/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_6Fill*ones_like_6/Shape/shape_as_tensor:output:0ones_like_6/Const:output:0*
T0*
_output_shapes
:M
mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �[
mul_13Mulones_like_6:output:0mul_13/y:output:0*
T0*
_output_shapes
:G
Log_3LogSelectV2_10:output:0*
T0*
_output_shapes
:P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_3RealDiv	Log_3:y:0truediv_3/y:output:0*
T0*
_output_shapes
:@
Neg_7Negtruediv_3:z:0*
T0*
_output_shapes
:D
Round_3Roundtruediv_3:z:0*
T0*
_output_shapes
:L
add_12AddV2	Neg_7:y:0Round_3:y:0*
T0*
_output_shapes
:O
StopGradient_5StopGradient
add_12:z:0*
T0*
_output_shapes
:\
add_13AddV2truediv_3:z:0StopGradient_5:output:0*
T0*
_output_shapes
:^
clip_by_value_3/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@w
clip_by_value_3/MinimumMinimum
add_13:z:0"clip_by_value_3/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0*
_output_shapes
:M
mul_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
mul_14Mulmul_14/x:output:0clip_by_value_3:z:0*
T0*
_output_shapes
:`
SelectV2_11SelectV2
Less_7:z:0
mul_13:z:0
mul_14:z:0*
T0*
_output_shapes
:L
pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_3Powpow_3/x:output:0SelectV2_11:output:0*
T0*
_output_shapes
:I
mul_15Mul
add_11:z:0	pow_3:z:0*
T0*
_output_shapes
:f
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource*
_output_shapes
:*
dtype0K
Neg_8NegReadVariableOp_7:value:0*
T0*
_output_shapes
:K
add_14AddV2	Neg_8:y:0
mul_15:z:0*
T0*
_output_shapes
:M
mul_16/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
mul_16Mulmul_16/x:output:0
add_14:z:0*
T0*
_output_shapes
:O
StopGradient_6StopGradient
mul_16:z:0*
T0*
_output_shapes
:f
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource*
_output_shapes
:*
dtype0g
add_15AddV2ReadVariableOp_8:value:0StopGradient_6:output:0*
T0*
_output_shapes
:p
Relu_3/ReadVariableOpReadVariableOprelu_3_readvariableop_resource*
_output_shapes
:*
dtype0R
Relu_3ReluRelu_3/ReadVariableOp:value:0*
T0*
_output_shapes
:p
Relu_4/ReadVariableOpReadVariableOprelu_3_readvariableop_resource*
_output_shapes
:*
dtype0R
Relu_4ReluRelu_4/ReadVariableOp:value:0*
T0*
_output_shapes
:M
Less_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3\
Less_8LessRelu_4:activations:0Less_8/y:output:0*
T0*
_output_shapes
:R
SelectV2_12/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3v
SelectV2_12SelectV2
Less_8:z:0SelectV2_12/t:output:0Relu_4:activations:0*
T0*
_output_shapes
:M
Less_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3\
Less_9LessRelu_4:activations:0Less_9/y:output:0*
T0*
_output_shapes
:k
!ones_like_7/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_7Fill*ones_like_7/Shape/shape_as_tensor:output:0ones_like_7/Const:output:0*
T0*
_output_shapes
:M
mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �[
mul_17Mulones_like_7:output:0mul_17/y:output:0*
T0*
_output_shapes
:G
SqrtSqrtSelectV2_12:output:0*
T0*
_output_shapes
:;
Log_4LogSqrt:y:0*
T0*
_output_shapes
:P
truediv_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_4RealDiv	Log_4:y:0truediv_4/y:output:0*
T0*
_output_shapes
:@
Neg_9Negtruediv_4:z:0*
T0*
_output_shapes
:D
Round_4Roundtruediv_4:z:0*
T0*
_output_shapes
:L
add_16AddV2	Neg_9:y:0Round_4:y:0*
T0*
_output_shapes
:O
StopGradient_7StopGradient
add_16:z:0*
T0*
_output_shapes
:\
add_17AddV2truediv_4:z:0StopGradient_7:output:0*
T0*
_output_shapes
:^
clip_by_value_4/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Aw
clip_by_value_4/MinimumMinimum
add_17:z:0"clip_by_value_4/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_4Maximumclip_by_value_4/Minimum:z:0clip_by_value_4/y:output:0*
T0*
_output_shapes
:M
mul_18/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Z
mul_18Mulmul_18/x:output:0clip_by_value_4:z:0*
T0*
_output_shapes
:`
SelectV2_13SelectV2
Less_9:z:0
mul_17:z:0
mul_18:z:0*
T0*
_output_shapes
:k
ReadVariableOp_9ReadVariableOprelu_3_readvariableop_resource*
_output_shapes
:*
dtype0L
Neg_10NegReadVariableOp_9:value:0*
T0*
_output_shapes
:?
Relu_5Relu
Neg_10:y:0*
T0*
_output_shapes
:M
mul_19/yConst*
_output_shapes
: *
dtype0*
valueB
 *    [
mul_19MulRelu_5:activations:0mul_19/y:output:0*
T0*
_output_shapes
:N
	Less_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3T
Less_10Less
mul_19:z:0Less_10/y:output:0*
T0*
_output_shapes
:R
SelectV2_14/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3m
SelectV2_14SelectV2Less_10:z:0SelectV2_14/t:output:0
mul_19:z:0*
T0*
_output_shapes
:N
	Less_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3T
Less_11Less
mul_19:z:0Less_11/y:output:0*
T0*
_output_shapes
:k
!ones_like_8/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:V
ones_like_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ones_like_8Fill*ones_like_8/Shape/shape_as_tensor:output:0ones_like_8/Const:output:0*
T0*
_output_shapes
:M
mul_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �[
mul_20Mulones_like_8:output:0mul_20/y:output:0*
T0*
_output_shapes
:I
Sqrt_1SqrtSelectV2_14:output:0*
T0*
_output_shapes
:=
Log_5Log
Sqrt_1:y:0*
T0*
_output_shapes
:P
truediv_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?Z
	truediv_5RealDiv	Log_5:y:0truediv_5/y:output:0*
T0*
_output_shapes
:A
Neg_11Negtruediv_5:z:0*
T0*
_output_shapes
:D
Round_5Roundtruediv_5:z:0*
T0*
_output_shapes
:M
add_18AddV2
Neg_11:y:0Round_5:y:0*
T0*
_output_shapes
:O
StopGradient_8StopGradient
add_18:z:0*
T0*
_output_shapes
:\
add_19AddV2truediv_5:z:0StopGradient_8:output:0*
T0*
_output_shapes
:^
clip_by_value_5/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Aw
clip_by_value_5/MinimumMinimum
add_19:z:0"clip_by_value_5/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_5Maximumclip_by_value_5/Minimum:z:0clip_by_value_5/y:output:0*
T0*
_output_shapes
:M
mul_21/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Z
mul_21Mulmul_21/x:output:0clip_by_value_5:z:0*
T0*
_output_shapes
:a
SelectV2_15SelectV2Less_11:z:0
mul_20:z:0
mul_21:z:0*
T0*
_output_shapes
:l
ReadVariableOp_10ReadVariableOprelu_3_readvariableop_resource*
_output_shapes
:*
dtype0U
GreaterEqual_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *    y
GreaterEqual_3GreaterEqualReadVariableOp_10:value:0GreaterEqual_3/y:output:0*
T0*
_output_shapes
:O
LogicalOr_1/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z`
LogicalOr_1	LogicalOrGreaterEqual_3:z:0LogicalOr_1/y:output:0*
_output_shapes
:L
pow_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_4Powpow_4/x:output:0SelectV2_13:output:0*
T0*
_output_shapes
:L
pow_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
pow_5Powpow_5/x:output:0SelectV2_15:output:0*
T0*
_output_shapes
:=
Neg_12Neg	pow_5:z:0*
T0*
_output_shapes
:d
SelectV2_16SelectV2LogicalOr_1:z:0	pow_4:z:0
Neg_12:y:0*
T0*
_output_shapes
:H
Neg_13NegRelu_3:activations:0*
T0*
_output_shapes
:V
add_20AddV2
Neg_13:y:0SelectV2_16:output:0*
T0*
_output_shapes
:M
mul_22/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
mul_22Mulmul_22/x:output:0
add_20:z:0*
T0*
_output_shapes
:O
StopGradient_9StopGradient
mul_22:z:0*
T0*
_output_shapes
:c
add_21AddV2Relu_3:activations:0StopGradient_9:output:0*
T0*
_output_shapes
:T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:a
batchnorm/addAddV2
add_21:z:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:Y
batchnorm/mulMulbatchnorm/Rsqrt:y:0	add_5:z:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������Z
batchnorm/mul_2Mul
add_15:z:0batchnorm/mul:z:0*
T0*
_output_shapes
:Z
batchnorm/subSub
add_10:z:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^Abs_1/ReadVariableOp^Abs_3/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_10^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8^ReadVariableOp_9^Relu/ReadVariableOp^Relu_1/ReadVariableOp^Relu_3/ReadVariableOp^Relu_4/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2,
Abs_1/ReadVariableOpAbs_1/ReadVariableOp2,
Abs_3/ReadVariableOpAbs_3/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12&
ReadVariableOp_10ReadVariableOp_102$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82$
ReadVariableOp_9ReadVariableOp_92*
Relu/ReadVariableOpRelu/ReadVariableOp2.
Relu_1/ReadVariableOpRelu_1/ReadVariableOp2.
Relu_3/ReadVariableOpRelu_3/ReadVariableOp2.
Relu_4/ReadVariableOpRelu_4/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�2
�
C__inference_dense1_layer_call_and_return_conditional_losses_8203214

inputs)
readvariableop_resource:, '
readvariableop_3_resource: 
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:, *
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   D[
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes

:, N
truedivRealDivmul:z:0Cast:y:0*
T0*
_output_shapes

:, @
NegNegtruediv:z:0*
T0*
_output_shapes

:, D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:, I
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:, N
StopGradientStopGradientadd:z:0*
T0*
_output_shapes

:, [
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:, \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ��Cv
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:, T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:, R
mul_1MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

:, P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   D^
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*
_output_shapes

:, L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*
_output_shapes

:, h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:, *
dtype0O
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes

:, M
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*
_output_shapes

:, L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*
_output_shapes

:, R
StopGradient_1StopGradient	mul_3:z:0*
T0*
_output_shapes

:, h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:, *
dtype0j
add_3AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*
_output_shapes

:, U
MatMulMatMulinputs	add_3:z:0*
T0*'
_output_shapes
:��������� I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
: *
dtype0L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   D]
mul_4MulReadVariableOp_3:value:0mul_4/y:output:0*
T0*
_output_shapes
: P
	truediv_2RealDiv	mul_4:z:0
Cast_1:y:0*
T0*
_output_shapes
: @
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
: D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
: K
add_4AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
: N
StopGradient_2StopGradient	add_4:z:0*
T0*
_output_shapes
: [
add_5AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
: ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ��Cv
clip_by_value_1/MinimumMinimum	add_5:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
: V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
: R
mul_5Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
: P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   DZ
	truediv_3RealDiv	mul_5:z:0truediv_3/y:output:0*
T0*
_output_shapes
: L
mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_6Mulmul_6/x:output:0truediv_3:z:0*
T0*
_output_shapes
: f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
: *
dtype0K
Neg_3NegReadVariableOp_4:value:0*
T0*
_output_shapes
: I
add_6AddV2	Neg_3:y:0	mul_6:z:0*
T0*
_output_shapes
: L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_7Mulmul_7/x:output:0	add_6:z:0*
T0*
_output_shapes
: N
StopGradient_3StopGradient	mul_7:z:0*
T0*
_output_shapes
: f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
: *
dtype0f
add_7AddV2ReadVariableOp_5:value:0StopGradient_3:output:0*
T0*
_output_shapes
: a
BiasAddBiasAddMatMul:product:0	add_7:z:0*
T0*'
_output_shapes
:��������� _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������,: : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_5:O K
'
_output_shapes
:���������,
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
��
�
"__inference__wrapped_model_8203144

inputsC
1simplified_encoder_dense1_readvariableop_resource:, A
3simplified_encoder_dense1_readvariableop_3_resource: <
.simplified_encoder_bn1_readvariableop_resource: >
0simplified_encoder_bn1_readvariableop_3_resource: >
0simplified_encoder_bn1_readvariableop_6_resource: C
5simplified_encoder_bn1_relu_3_readvariableop_resource: C
1simplified_encoder_dense2_readvariableop_resource: A
3simplified_encoder_dense2_readvariableop_3_resource:<
.simplified_encoder_bn2_readvariableop_resource:>
0simplified_encoder_bn2_readvariableop_3_resource:>
0simplified_encoder_bn2_readvariableop_6_resource:C
5simplified_encoder_bn2_relu_3_readvariableop_resource:C
1simplified_encoder_z_mean_readvariableop_resource:A
3simplified_encoder_z_mean_readvariableop_3_resource:
identity��+simplified_encoder/BN1/Abs_1/ReadVariableOp�+simplified_encoder/BN1/Abs_3/ReadVariableOp�%simplified_encoder/BN1/ReadVariableOp�'simplified_encoder/BN1/ReadVariableOp_1�(simplified_encoder/BN1/ReadVariableOp_10�'simplified_encoder/BN1/ReadVariableOp_2�'simplified_encoder/BN1/ReadVariableOp_3�'simplified_encoder/BN1/ReadVariableOp_4�'simplified_encoder/BN1/ReadVariableOp_5�'simplified_encoder/BN1/ReadVariableOp_6�'simplified_encoder/BN1/ReadVariableOp_7�'simplified_encoder/BN1/ReadVariableOp_8�'simplified_encoder/BN1/ReadVariableOp_9�*simplified_encoder/BN1/Relu/ReadVariableOp�,simplified_encoder/BN1/Relu_1/ReadVariableOp�,simplified_encoder/BN1/Relu_3/ReadVariableOp�,simplified_encoder/BN1/Relu_4/ReadVariableOp�+simplified_encoder/BN2/Abs_1/ReadVariableOp�+simplified_encoder/BN2/Abs_3/ReadVariableOp�%simplified_encoder/BN2/ReadVariableOp�'simplified_encoder/BN2/ReadVariableOp_1�(simplified_encoder/BN2/ReadVariableOp_10�'simplified_encoder/BN2/ReadVariableOp_2�'simplified_encoder/BN2/ReadVariableOp_3�'simplified_encoder/BN2/ReadVariableOp_4�'simplified_encoder/BN2/ReadVariableOp_5�'simplified_encoder/BN2/ReadVariableOp_6�'simplified_encoder/BN2/ReadVariableOp_7�'simplified_encoder/BN2/ReadVariableOp_8�'simplified_encoder/BN2/ReadVariableOp_9�*simplified_encoder/BN2/Relu/ReadVariableOp�,simplified_encoder/BN2/Relu_1/ReadVariableOp�,simplified_encoder/BN2/Relu_3/ReadVariableOp�,simplified_encoder/BN2/Relu_4/ReadVariableOp�(simplified_encoder/dense1/ReadVariableOp�*simplified_encoder/dense1/ReadVariableOp_1�*simplified_encoder/dense1/ReadVariableOp_2�*simplified_encoder/dense1/ReadVariableOp_3�*simplified_encoder/dense1/ReadVariableOp_4�*simplified_encoder/dense1/ReadVariableOp_5�(simplified_encoder/dense2/ReadVariableOp�*simplified_encoder/dense2/ReadVariableOp_1�*simplified_encoder/dense2/ReadVariableOp_2�*simplified_encoder/dense2/ReadVariableOp_3�*simplified_encoder/dense2/ReadVariableOp_4�*simplified_encoder/dense2/ReadVariableOp_5�(simplified_encoder/z_mean/ReadVariableOp�*simplified_encoder/z_mean/ReadVariableOp_1�*simplified_encoder/z_mean/ReadVariableOp_2�*simplified_encoder/z_mean/ReadVariableOp_3�*simplified_encoder/z_mean/ReadVariableOp_4�*simplified_encoder/z_mean/ReadVariableOp_5a
simplified_encoder/dense1/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :a
simplified_encoder/dense1/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :�
simplified_encoder/dense1/PowPow(simplified_encoder/dense1/Pow/x:output:0(simplified_encoder/dense1/Pow/y:output:0*
T0*
_output_shapes
: y
simplified_encoder/dense1/CastCast!simplified_encoder/dense1/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: �
(simplified_encoder/dense1/ReadVariableOpReadVariableOp1simplified_encoder_dense1_readvariableop_resource*
_output_shapes

:, *
dtype0d
simplified_encoder/dense1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   D�
simplified_encoder/dense1/mulMul0simplified_encoder/dense1/ReadVariableOp:value:0(simplified_encoder/dense1/mul/y:output:0*
T0*
_output_shapes

:, �
!simplified_encoder/dense1/truedivRealDiv!simplified_encoder/dense1/mul:z:0"simplified_encoder/dense1/Cast:y:0*
T0*
_output_shapes

:, t
simplified_encoder/dense1/NegNeg%simplified_encoder/dense1/truediv:z:0*
T0*
_output_shapes

:, x
simplified_encoder/dense1/RoundRound%simplified_encoder/dense1/truediv:z:0*
T0*
_output_shapes

:, �
simplified_encoder/dense1/addAddV2!simplified_encoder/dense1/Neg:y:0#simplified_encoder/dense1/Round:y:0*
T0*
_output_shapes

:, �
&simplified_encoder/dense1/StopGradientStopGradient!simplified_encoder/dense1/add:z:0*
T0*
_output_shapes

:, �
simplified_encoder/dense1/add_1AddV2%simplified_encoder/dense1/truediv:z:0/simplified_encoder/dense1/StopGradient:output:0*
T0*
_output_shapes

:, v
1simplified_encoder/dense1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ��C�
/simplified_encoder/dense1/clip_by_value/MinimumMinimum#simplified_encoder/dense1/add_1:z:0:simplified_encoder/dense1/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:, n
)simplified_encoder/dense1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
'simplified_encoder/dense1/clip_by_valueMaximum3simplified_encoder/dense1/clip_by_value/Minimum:z:02simplified_encoder/dense1/clip_by_value/y:output:0*
T0*
_output_shapes

:, �
simplified_encoder/dense1/mul_1Mul"simplified_encoder/dense1/Cast:y:0+simplified_encoder/dense1/clip_by_value:z:0*
T0*
_output_shapes

:, j
%simplified_encoder/dense1/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   D�
#simplified_encoder/dense1/truediv_1RealDiv#simplified_encoder/dense1/mul_1:z:0.simplified_encoder/dense1/truediv_1/y:output:0*
T0*
_output_shapes

:, f
!simplified_encoder/dense1/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/dense1/mul_2Mul*simplified_encoder/dense1/mul_2/x:output:0'simplified_encoder/dense1/truediv_1:z:0*
T0*
_output_shapes

:, �
*simplified_encoder/dense1/ReadVariableOp_1ReadVariableOp1simplified_encoder_dense1_readvariableop_resource*
_output_shapes

:, *
dtype0�
simplified_encoder/dense1/Neg_1Neg2simplified_encoder/dense1/ReadVariableOp_1:value:0*
T0*
_output_shapes

:, �
simplified_encoder/dense1/add_2AddV2#simplified_encoder/dense1/Neg_1:y:0#simplified_encoder/dense1/mul_2:z:0*
T0*
_output_shapes

:, f
!simplified_encoder/dense1/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/dense1/mul_3Mul*simplified_encoder/dense1/mul_3/x:output:0#simplified_encoder/dense1/add_2:z:0*
T0*
_output_shapes

:, �
(simplified_encoder/dense1/StopGradient_1StopGradient#simplified_encoder/dense1/mul_3:z:0*
T0*
_output_shapes

:, �
*simplified_encoder/dense1/ReadVariableOp_2ReadVariableOp1simplified_encoder_dense1_readvariableop_resource*
_output_shapes

:, *
dtype0�
simplified_encoder/dense1/add_3AddV22simplified_encoder/dense1/ReadVariableOp_2:value:01simplified_encoder/dense1/StopGradient_1:output:0*
T0*
_output_shapes

:, �
 simplified_encoder/dense1/MatMulMatMulinputs#simplified_encoder/dense1/add_3:z:0*
T0*'
_output_shapes
:��������� c
!simplified_encoder/dense1/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :c
!simplified_encoder/dense1/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
simplified_encoder/dense1/Pow_1Pow*simplified_encoder/dense1/Pow_1/x:output:0*simplified_encoder/dense1/Pow_1/y:output:0*
T0*
_output_shapes
: }
 simplified_encoder/dense1/Cast_1Cast#simplified_encoder/dense1/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
*simplified_encoder/dense1/ReadVariableOp_3ReadVariableOp3simplified_encoder_dense1_readvariableop_3_resource*
_output_shapes
: *
dtype0f
!simplified_encoder/dense1/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   D�
simplified_encoder/dense1/mul_4Mul2simplified_encoder/dense1/ReadVariableOp_3:value:0*simplified_encoder/dense1/mul_4/y:output:0*
T0*
_output_shapes
: �
#simplified_encoder/dense1/truediv_2RealDiv#simplified_encoder/dense1/mul_4:z:0$simplified_encoder/dense1/Cast_1:y:0*
T0*
_output_shapes
: t
simplified_encoder/dense1/Neg_2Neg'simplified_encoder/dense1/truediv_2:z:0*
T0*
_output_shapes
: x
!simplified_encoder/dense1/Round_1Round'simplified_encoder/dense1/truediv_2:z:0*
T0*
_output_shapes
: �
simplified_encoder/dense1/add_4AddV2#simplified_encoder/dense1/Neg_2:y:0%simplified_encoder/dense1/Round_1:y:0*
T0*
_output_shapes
: �
(simplified_encoder/dense1/StopGradient_2StopGradient#simplified_encoder/dense1/add_4:z:0*
T0*
_output_shapes
: �
simplified_encoder/dense1/add_5AddV2'simplified_encoder/dense1/truediv_2:z:01simplified_encoder/dense1/StopGradient_2:output:0*
T0*
_output_shapes
: x
3simplified_encoder/dense1/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ��C�
1simplified_encoder/dense1/clip_by_value_1/MinimumMinimum#simplified_encoder/dense1/add_5:z:0<simplified_encoder/dense1/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
: p
+simplified_encoder/dense1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
)simplified_encoder/dense1/clip_by_value_1Maximum5simplified_encoder/dense1/clip_by_value_1/Minimum:z:04simplified_encoder/dense1/clip_by_value_1/y:output:0*
T0*
_output_shapes
: �
simplified_encoder/dense1/mul_5Mul$simplified_encoder/dense1/Cast_1:y:0-simplified_encoder/dense1/clip_by_value_1:z:0*
T0*
_output_shapes
: j
%simplified_encoder/dense1/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   D�
#simplified_encoder/dense1/truediv_3RealDiv#simplified_encoder/dense1/mul_5:z:0.simplified_encoder/dense1/truediv_3/y:output:0*
T0*
_output_shapes
: f
!simplified_encoder/dense1/mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/dense1/mul_6Mul*simplified_encoder/dense1/mul_6/x:output:0'simplified_encoder/dense1/truediv_3:z:0*
T0*
_output_shapes
: �
*simplified_encoder/dense1/ReadVariableOp_4ReadVariableOp3simplified_encoder_dense1_readvariableop_3_resource*
_output_shapes
: *
dtype0
simplified_encoder/dense1/Neg_3Neg2simplified_encoder/dense1/ReadVariableOp_4:value:0*
T0*
_output_shapes
: �
simplified_encoder/dense1/add_6AddV2#simplified_encoder/dense1/Neg_3:y:0#simplified_encoder/dense1/mul_6:z:0*
T0*
_output_shapes
: f
!simplified_encoder/dense1/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/dense1/mul_7Mul*simplified_encoder/dense1/mul_7/x:output:0#simplified_encoder/dense1/add_6:z:0*
T0*
_output_shapes
: �
(simplified_encoder/dense1/StopGradient_3StopGradient#simplified_encoder/dense1/mul_7:z:0*
T0*
_output_shapes
: �
*simplified_encoder/dense1/ReadVariableOp_5ReadVariableOp3simplified_encoder_dense1_readvariableop_3_resource*
_output_shapes
: *
dtype0�
simplified_encoder/dense1/add_7AddV22simplified_encoder/dense1/ReadVariableOp_5:value:01simplified_encoder/dense1/StopGradient_3:output:0*
T0*
_output_shapes
: �
!simplified_encoder/dense1/BiasAddBiasAdd*simplified_encoder/dense1/MatMul:product:0#simplified_encoder/dense1/add_7:z:0*
T0*'
_output_shapes
:��������� �
%simplified_encoder/BN1/ReadVariableOpReadVariableOp.simplified_encoder_bn1_readvariableop_resource*
_output_shapes
: *
dtype0g
"simplified_encoder/BN1/LessEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   E�
 simplified_encoder/BN1/LessEqual	LessEqual-simplified_encoder/BN1/ReadVariableOp:value:0+simplified_encoder/BN1/LessEqual/y:output:0*
T0*
_output_shapes
: �
*simplified_encoder/BN1/Relu/ReadVariableOpReadVariableOp.simplified_encoder_bn1_readvariableop_resource*
_output_shapes
: *
dtype0|
simplified_encoder/BN1/ReluRelu2simplified_encoder/BN1/Relu/ReadVariableOp:value:0*
T0*
_output_shapes
: �
/simplified_encoder/BN1/ones_like/ReadVariableOpReadVariableOp.simplified_encoder_bn1_readvariableop_resource*
_output_shapes
: *
dtype0�
6simplified_encoder/BN1/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: k
&simplified_encoder/BN1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
 simplified_encoder/BN1/ones_likeFill?simplified_encoder/BN1/ones_like/Shape/shape_as_tensor:output:0/simplified_encoder/BN1/ones_like/Const:output:0*
T0*
_output_shapes
: a
simplified_encoder/BN1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   E�
simplified_encoder/BN1/mulMul)simplified_encoder/BN1/ones_like:output:0%simplified_encoder/BN1/mul/y:output:0*
T0*
_output_shapes
: �
simplified_encoder/BN1/SelectV2SelectV2$simplified_encoder/BN1/LessEqual:z:0)simplified_encoder/BN1/Relu:activations:0simplified_encoder/BN1/mul:z:0*
T0*
_output_shapes
: �
,simplified_encoder/BN1/Relu_1/ReadVariableOpReadVariableOp.simplified_encoder_bn1_readvariableop_resource*
_output_shapes
: *
dtype0�
simplified_encoder/BN1/Relu_1Relu4simplified_encoder/BN1/Relu_1/ReadVariableOp:value:0*
T0*
_output_shapes
: b
simplified_encoder/BN1/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
simplified_encoder/BN1/LessLess+simplified_encoder/BN1/Relu_1:activations:0&simplified_encoder/BN1/Less/y:output:0*
T0*
_output_shapes
: h
#simplified_encoder/BN1/SelectV2_1/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
!simplified_encoder/BN1/SelectV2_1SelectV2simplified_encoder/BN1/Less:z:0,simplified_encoder/BN1/SelectV2_1/t:output:0+simplified_encoder/BN1/Relu_1:activations:0*
T0*
_output_shapes
: j
%simplified_encoder/BN1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   E�
#simplified_encoder/BN1/GreaterEqualGreaterEqual*simplified_encoder/BN1/SelectV2_1:output:0.simplified_encoder/BN1/GreaterEqual/y:output:0*
T0*
_output_shapes
: �
8simplified_encoder/BN1/ones_like_1/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: m
(simplified_encoder/BN1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"simplified_encoder/BN1/ones_like_1FillAsimplified_encoder/BN1/ones_like_1/Shape/shape_as_tensor:output:01simplified_encoder/BN1/ones_like_1/Const:output:0*
T0*
_output_shapes
: c
simplified_encoder/BN1/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   E�
simplified_encoder/BN1/mul_1Mul+simplified_encoder/BN1/ones_like_1:output:0'simplified_encoder/BN1/mul_1/y:output:0*
T0*
_output_shapes
: �
!simplified_encoder/BN1/SelectV2_2SelectV2'simplified_encoder/BN1/GreaterEqual:z:0 simplified_encoder/BN1/mul_1:z:0*simplified_encoder/BN1/SelectV2_1:output:0*
T0*
_output_shapes
: d
simplified_encoder/BN1/Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
simplified_encoder/BN1/Less_1Less+simplified_encoder/BN1/Relu_1:activations:0(simplified_encoder/BN1/Less_1/y:output:0*
T0*
_output_shapes
: �
8simplified_encoder/BN1/ones_like_2/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: m
(simplified_encoder/BN1/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"simplified_encoder/BN1/ones_like_2FillAsimplified_encoder/BN1/ones_like_2/Shape/shape_as_tensor:output:01simplified_encoder/BN1/ones_like_2/Const:output:0*
T0*
_output_shapes
: c
simplified_encoder/BN1/mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
simplified_encoder/BN1/mul_2Mul+simplified_encoder/BN1/ones_like_2:output:0'simplified_encoder/BN1/mul_2/y:output:0*
T0*
_output_shapes
: r
simplified_encoder/BN1/LogLog*simplified_encoder/BN1/SelectV2_2:output:0*
T0*
_output_shapes
: e
 simplified_encoder/BN1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?�
simplified_encoder/BN1/truedivRealDivsimplified_encoder/BN1/Log:y:0)simplified_encoder/BN1/truediv/y:output:0*
T0*
_output_shapes
: j
simplified_encoder/BN1/NegNeg"simplified_encoder/BN1/truediv:z:0*
T0*
_output_shapes
: n
simplified_encoder/BN1/RoundRound"simplified_encoder/BN1/truediv:z:0*
T0*
_output_shapes
: �
simplified_encoder/BN1/addAddV2simplified_encoder/BN1/Neg:y:0 simplified_encoder/BN1/Round:y:0*
T0*
_output_shapes
: x
#simplified_encoder/BN1/StopGradientStopGradientsimplified_encoder/BN1/add:z:0*
T0*
_output_shapes
: �
simplified_encoder/BN1/add_1AddV2"simplified_encoder/BN1/truediv:z:0,simplified_encoder/BN1/StopGradient:output:0*
T0*
_output_shapes
: s
.simplified_encoder/BN1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
,simplified_encoder/BN1/clip_by_value/MinimumMinimum simplified_encoder/BN1/add_1:z:07simplified_encoder/BN1/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
: k
&simplified_encoder/BN1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
$simplified_encoder/BN1/clip_by_valueMaximum0simplified_encoder/BN1/clip_by_value/Minimum:z:0/simplified_encoder/BN1/clip_by_value/y:output:0*
T0*
_output_shapes
: c
simplified_encoder/BN1/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/BN1/mul_3Mul'simplified_encoder/BN1/mul_3/x:output:0(simplified_encoder/BN1/clip_by_value:z:0*
T0*
_output_shapes
: �
!simplified_encoder/BN1/SelectV2_3SelectV2!simplified_encoder/BN1/Less_1:z:0 simplified_encoder/BN1/mul_2:z:0 simplified_encoder/BN1/mul_3:z:0*
T0*
_output_shapes
: �
'simplified_encoder/BN1/ReadVariableOp_1ReadVariableOp.simplified_encoder_bn1_readvariableop_resource*
_output_shapes
: *
dtype0y
simplified_encoder/BN1/Neg_1Neg/simplified_encoder/BN1/ReadVariableOp_1:value:0*
T0*
_output_shapes
: l
simplified_encoder/BN1/Relu_2Relu simplified_encoder/BN1/Neg_1:y:0*
T0*
_output_shapes
: c
simplified_encoder/BN1/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
simplified_encoder/BN1/mul_4Mul+simplified_encoder/BN1/Relu_2:activations:0'simplified_encoder/BN1/mul_4/y:output:0*
T0*
_output_shapes
: d
simplified_encoder/BN1/Less_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
simplified_encoder/BN1/Less_2Less simplified_encoder/BN1/mul_4:z:0(simplified_encoder/BN1/Less_2/y:output:0*
T0*
_output_shapes
: h
#simplified_encoder/BN1/SelectV2_4/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
!simplified_encoder/BN1/SelectV2_4SelectV2!simplified_encoder/BN1/Less_2:z:0,simplified_encoder/BN1/SelectV2_4/t:output:0 simplified_encoder/BN1/mul_4:z:0*
T0*
_output_shapes
: l
'simplified_encoder/BN1/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   E�
%simplified_encoder/BN1/GreaterEqual_1GreaterEqual*simplified_encoder/BN1/SelectV2_4:output:00simplified_encoder/BN1/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: �
8simplified_encoder/BN1/ones_like_3/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: m
(simplified_encoder/BN1/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"simplified_encoder/BN1/ones_like_3FillAsimplified_encoder/BN1/ones_like_3/Shape/shape_as_tensor:output:01simplified_encoder/BN1/ones_like_3/Const:output:0*
T0*
_output_shapes
: c
simplified_encoder/BN1/mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *   E�
simplified_encoder/BN1/mul_5Mul+simplified_encoder/BN1/ones_like_3:output:0'simplified_encoder/BN1/mul_5/y:output:0*
T0*
_output_shapes
: �
!simplified_encoder/BN1/SelectV2_5SelectV2)simplified_encoder/BN1/GreaterEqual_1:z:0 simplified_encoder/BN1/mul_5:z:0*simplified_encoder/BN1/SelectV2_4:output:0*
T0*
_output_shapes
: d
simplified_encoder/BN1/Less_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
simplified_encoder/BN1/Less_3Less simplified_encoder/BN1/mul_4:z:0(simplified_encoder/BN1/Less_3/y:output:0*
T0*
_output_shapes
: �
8simplified_encoder/BN1/ones_like_4/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: m
(simplified_encoder/BN1/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"simplified_encoder/BN1/ones_like_4FillAsimplified_encoder/BN1/ones_like_4/Shape/shape_as_tensor:output:01simplified_encoder/BN1/ones_like_4/Const:output:0*
T0*
_output_shapes
: c
simplified_encoder/BN1/mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
simplified_encoder/BN1/mul_6Mul+simplified_encoder/BN1/ones_like_4:output:0'simplified_encoder/BN1/mul_6/y:output:0*
T0*
_output_shapes
: t
simplified_encoder/BN1/Log_1Log*simplified_encoder/BN1/SelectV2_5:output:0*
T0*
_output_shapes
: g
"simplified_encoder/BN1/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?�
 simplified_encoder/BN1/truediv_1RealDiv simplified_encoder/BN1/Log_1:y:0+simplified_encoder/BN1/truediv_1/y:output:0*
T0*
_output_shapes
: n
simplified_encoder/BN1/Neg_2Neg$simplified_encoder/BN1/truediv_1:z:0*
T0*
_output_shapes
: r
simplified_encoder/BN1/Round_1Round$simplified_encoder/BN1/truediv_1:z:0*
T0*
_output_shapes
: �
simplified_encoder/BN1/add_2AddV2 simplified_encoder/BN1/Neg_2:y:0"simplified_encoder/BN1/Round_1:y:0*
T0*
_output_shapes
: |
%simplified_encoder/BN1/StopGradient_1StopGradient simplified_encoder/BN1/add_2:z:0*
T0*
_output_shapes
: �
simplified_encoder/BN1/add_3AddV2$simplified_encoder/BN1/truediv_1:z:0.simplified_encoder/BN1/StopGradient_1:output:0*
T0*
_output_shapes
: u
0simplified_encoder/BN1/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
.simplified_encoder/BN1/clip_by_value_1/MinimumMinimum simplified_encoder/BN1/add_3:z:09simplified_encoder/BN1/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
: m
(simplified_encoder/BN1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
&simplified_encoder/BN1/clip_by_value_1Maximum2simplified_encoder/BN1/clip_by_value_1/Minimum:z:01simplified_encoder/BN1/clip_by_value_1/y:output:0*
T0*
_output_shapes
: c
simplified_encoder/BN1/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/BN1/mul_7Mul'simplified_encoder/BN1/mul_7/x:output:0*simplified_encoder/BN1/clip_by_value_1:z:0*
T0*
_output_shapes
: �
!simplified_encoder/BN1/SelectV2_6SelectV2!simplified_encoder/BN1/Less_3:z:0 simplified_encoder/BN1/mul_6:z:0 simplified_encoder/BN1/mul_7:z:0*
T0*
_output_shapes
: �
'simplified_encoder/BN1/ReadVariableOp_2ReadVariableOp.simplified_encoder_bn1_readvariableop_resource*
_output_shapes
: *
dtype0l
'simplified_encoder/BN1/GreaterEqual_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
%simplified_encoder/BN1/GreaterEqual_2GreaterEqual/simplified_encoder/BN1/ReadVariableOp_2:value:00simplified_encoder/BN1/GreaterEqual_2/y:output:0*
T0*
_output_shapes
: d
"simplified_encoder/BN1/LogicalOr/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z�
 simplified_encoder/BN1/LogicalOr	LogicalOr)simplified_encoder/BN1/GreaterEqual_2:z:0+simplified_encoder/BN1/LogicalOr/y:output:0*
_output_shapes
: a
simplified_encoder/BN1/pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
simplified_encoder/BN1/powPow%simplified_encoder/BN1/pow/x:output:0*simplified_encoder/BN1/SelectV2_3:output:0*
T0*
_output_shapes
: c
simplified_encoder/BN1/pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
simplified_encoder/BN1/pow_1Pow'simplified_encoder/BN1/pow_1/x:output:0*simplified_encoder/BN1/SelectV2_6:output:0*
T0*
_output_shapes
: j
simplified_encoder/BN1/Neg_3Neg simplified_encoder/BN1/pow_1:z:0*
T0*
_output_shapes
: �
!simplified_encoder/BN1/SelectV2_7SelectV2$simplified_encoder/BN1/LogicalOr:z:0simplified_encoder/BN1/pow:z:0 simplified_encoder/BN1/Neg_3:y:0*
T0*
_output_shapes
: r
simplified_encoder/BN1/Neg_4Neg(simplified_encoder/BN1/SelectV2:output:0*
T0*
_output_shapes
: �
simplified_encoder/BN1/add_4AddV2 simplified_encoder/BN1/Neg_4:y:0*simplified_encoder/BN1/SelectV2_7:output:0*
T0*
_output_shapes
: c
simplified_encoder/BN1/mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/BN1/mul_8Mul'simplified_encoder/BN1/mul_8/x:output:0 simplified_encoder/BN1/add_4:z:0*
T0*
_output_shapes
: |
%simplified_encoder/BN1/StopGradient_2StopGradient simplified_encoder/BN1/mul_8:z:0*
T0*
_output_shapes
: �
simplified_encoder/BN1/add_5AddV2(simplified_encoder/BN1/SelectV2:output:0.simplified_encoder/BN1/StopGradient_2:output:0*
T0*
_output_shapes
: �
'simplified_encoder/BN1/ReadVariableOp_3ReadVariableOp0simplified_encoder_bn1_readvariableop_3_resource*
_output_shapes
: *
dtype0y
simplified_encoder/BN1/SignSign/simplified_encoder/BN1/ReadVariableOp_3:value:0*
T0*
_output_shapes
: g
simplified_encoder/BN1/AbsAbssimplified_encoder/BN1/Sign:y:0*
T0*
_output_shapes
: a
simplified_encoder/BN1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/BN1/subSub%simplified_encoder/BN1/sub/x:output:0simplified_encoder/BN1/Abs:y:0*
T0*
_output_shapes
: �
simplified_encoder/BN1/add_6AddV2simplified_encoder/BN1/Sign:y:0simplified_encoder/BN1/sub:z:0*
T0*
_output_shapes
: �
+simplified_encoder/BN1/Abs_1/ReadVariableOpReadVariableOp0simplified_encoder_bn1_readvariableop_3_resource*
_output_shapes
: *
dtype0}
simplified_encoder/BN1/Abs_1Abs3simplified_encoder/BN1/Abs_1/ReadVariableOp:value:0*
T0*
_output_shapes
: d
simplified_encoder/BN1/Less_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
simplified_encoder/BN1/Less_4Less simplified_encoder/BN1/Abs_1:y:0(simplified_encoder/BN1/Less_4/y:output:0*
T0*
_output_shapes
: h
#simplified_encoder/BN1/SelectV2_8/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
!simplified_encoder/BN1/SelectV2_8SelectV2!simplified_encoder/BN1/Less_4:z:0,simplified_encoder/BN1/SelectV2_8/t:output:0 simplified_encoder/BN1/Abs_1:y:0*
T0*
_output_shapes
: d
simplified_encoder/BN1/Less_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
simplified_encoder/BN1/Less_5Less simplified_encoder/BN1/Abs_1:y:0(simplified_encoder/BN1/Less_5/y:output:0*
T0*
_output_shapes
: �
8simplified_encoder/BN1/ones_like_5/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: m
(simplified_encoder/BN1/ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"simplified_encoder/BN1/ones_like_5FillAsimplified_encoder/BN1/ones_like_5/Shape/shape_as_tensor:output:01simplified_encoder/BN1/ones_like_5/Const:output:0*
T0*
_output_shapes
: c
simplified_encoder/BN1/mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
simplified_encoder/BN1/mul_9Mul+simplified_encoder/BN1/ones_like_5:output:0'simplified_encoder/BN1/mul_9/y:output:0*
T0*
_output_shapes
: t
simplified_encoder/BN1/Log_2Log*simplified_encoder/BN1/SelectV2_8:output:0*
T0*
_output_shapes
: g
"simplified_encoder/BN1/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?�
 simplified_encoder/BN1/truediv_2RealDiv simplified_encoder/BN1/Log_2:y:0+simplified_encoder/BN1/truediv_2/y:output:0*
T0*
_output_shapes
: n
simplified_encoder/BN1/Neg_5Neg$simplified_encoder/BN1/truediv_2:z:0*
T0*
_output_shapes
: r
simplified_encoder/BN1/Round_2Round$simplified_encoder/BN1/truediv_2:z:0*
T0*
_output_shapes
: �
simplified_encoder/BN1/add_7AddV2 simplified_encoder/BN1/Neg_5:y:0"simplified_encoder/BN1/Round_2:y:0*
T0*
_output_shapes
: |
%simplified_encoder/BN1/StopGradient_3StopGradient simplified_encoder/BN1/add_7:z:0*
T0*
_output_shapes
: �
simplified_encoder/BN1/add_8AddV2$simplified_encoder/BN1/truediv_2:z:0.simplified_encoder/BN1/StopGradient_3:output:0*
T0*
_output_shapes
: u
0simplified_encoder/BN1/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@�
.simplified_encoder/BN1/clip_by_value_2/MinimumMinimum simplified_encoder/BN1/add_8:z:09simplified_encoder/BN1/clip_by_value_2/Minimum/y:output:0*
T0*
_output_shapes
: m
(simplified_encoder/BN1/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
&simplified_encoder/BN1/clip_by_value_2Maximum2simplified_encoder/BN1/clip_by_value_2/Minimum:z:01simplified_encoder/BN1/clip_by_value_2/y:output:0*
T0*
_output_shapes
: d
simplified_encoder/BN1/mul_10/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/BN1/mul_10Mul(simplified_encoder/BN1/mul_10/x:output:0*simplified_encoder/BN1/clip_by_value_2:z:0*
T0*
_output_shapes
: �
!simplified_encoder/BN1/SelectV2_9SelectV2!simplified_encoder/BN1/Less_5:z:0 simplified_encoder/BN1/mul_9:z:0!simplified_encoder/BN1/mul_10:z:0*
T0*
_output_shapes
: c
simplified_encoder/BN1/pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
simplified_encoder/BN1/pow_2Pow'simplified_encoder/BN1/pow_2/x:output:0*simplified_encoder/BN1/SelectV2_9:output:0*
T0*
_output_shapes
: �
simplified_encoder/BN1/mul_11Mul simplified_encoder/BN1/add_6:z:0 simplified_encoder/BN1/pow_2:z:0*
T0*
_output_shapes
: �
'simplified_encoder/BN1/ReadVariableOp_4ReadVariableOp0simplified_encoder_bn1_readvariableop_3_resource*
_output_shapes
: *
dtype0y
simplified_encoder/BN1/Neg_6Neg/simplified_encoder/BN1/ReadVariableOp_4:value:0*
T0*
_output_shapes
: �
simplified_encoder/BN1/add_9AddV2 simplified_encoder/BN1/Neg_6:y:0!simplified_encoder/BN1/mul_11:z:0*
T0*
_output_shapes
: d
simplified_encoder/BN1/mul_12/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/BN1/mul_12Mul(simplified_encoder/BN1/mul_12/x:output:0 simplified_encoder/BN1/add_9:z:0*
T0*
_output_shapes
: }
%simplified_encoder/BN1/StopGradient_4StopGradient!simplified_encoder/BN1/mul_12:z:0*
T0*
_output_shapes
: �
'simplified_encoder/BN1/ReadVariableOp_5ReadVariableOp0simplified_encoder_bn1_readvariableop_3_resource*
_output_shapes
: *
dtype0�
simplified_encoder/BN1/add_10AddV2/simplified_encoder/BN1/ReadVariableOp_5:value:0.simplified_encoder/BN1/StopGradient_4:output:0*
T0*
_output_shapes
: �
'simplified_encoder/BN1/ReadVariableOp_6ReadVariableOp0simplified_encoder_bn1_readvariableop_6_resource*
_output_shapes
: *
dtype0{
simplified_encoder/BN1/Sign_1Sign/simplified_encoder/BN1/ReadVariableOp_6:value:0*
T0*
_output_shapes
: k
simplified_encoder/BN1/Abs_2Abs!simplified_encoder/BN1/Sign_1:y:0*
T0*
_output_shapes
: c
simplified_encoder/BN1/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/BN1/sub_1Sub'simplified_encoder/BN1/sub_1/x:output:0 simplified_encoder/BN1/Abs_2:y:0*
T0*
_output_shapes
: �
simplified_encoder/BN1/add_11AddV2!simplified_encoder/BN1/Sign_1:y:0 simplified_encoder/BN1/sub_1:z:0*
T0*
_output_shapes
: �
+simplified_encoder/BN1/Abs_3/ReadVariableOpReadVariableOp0simplified_encoder_bn1_readvariableop_6_resource*
_output_shapes
: *
dtype0}
simplified_encoder/BN1/Abs_3Abs3simplified_encoder/BN1/Abs_3/ReadVariableOp:value:0*
T0*
_output_shapes
: d
simplified_encoder/BN1/Less_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
simplified_encoder/BN1/Less_6Less simplified_encoder/BN1/Abs_3:y:0(simplified_encoder/BN1/Less_6/y:output:0*
T0*
_output_shapes
: i
$simplified_encoder/BN1/SelectV2_10/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
"simplified_encoder/BN1/SelectV2_10SelectV2!simplified_encoder/BN1/Less_6:z:0-simplified_encoder/BN1/SelectV2_10/t:output:0 simplified_encoder/BN1/Abs_3:y:0*
T0*
_output_shapes
: d
simplified_encoder/BN1/Less_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
simplified_encoder/BN1/Less_7Less simplified_encoder/BN1/Abs_3:y:0(simplified_encoder/BN1/Less_7/y:output:0*
T0*
_output_shapes
: �
8simplified_encoder/BN1/ones_like_6/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: m
(simplified_encoder/BN1/ones_like_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"simplified_encoder/BN1/ones_like_6FillAsimplified_encoder/BN1/ones_like_6/Shape/shape_as_tensor:output:01simplified_encoder/BN1/ones_like_6/Const:output:0*
T0*
_output_shapes
: d
simplified_encoder/BN1/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
simplified_encoder/BN1/mul_13Mul+simplified_encoder/BN1/ones_like_6:output:0(simplified_encoder/BN1/mul_13/y:output:0*
T0*
_output_shapes
: u
simplified_encoder/BN1/Log_3Log+simplified_encoder/BN1/SelectV2_10:output:0*
T0*
_output_shapes
: g
"simplified_encoder/BN1/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?�
 simplified_encoder/BN1/truediv_3RealDiv simplified_encoder/BN1/Log_3:y:0+simplified_encoder/BN1/truediv_3/y:output:0*
T0*
_output_shapes
: n
simplified_encoder/BN1/Neg_7Neg$simplified_encoder/BN1/truediv_3:z:0*
T0*
_output_shapes
: r
simplified_encoder/BN1/Round_3Round$simplified_encoder/BN1/truediv_3:z:0*
T0*
_output_shapes
: �
simplified_encoder/BN1/add_12AddV2 simplified_encoder/BN1/Neg_7:y:0"simplified_encoder/BN1/Round_3:y:0*
T0*
_output_shapes
: }
%simplified_encoder/BN1/StopGradient_5StopGradient!simplified_encoder/BN1/add_12:z:0*
T0*
_output_shapes
: �
simplified_encoder/BN1/add_13AddV2$simplified_encoder/BN1/truediv_3:z:0.simplified_encoder/BN1/StopGradient_5:output:0*
T0*
_output_shapes
: u
0simplified_encoder/BN1/clip_by_value_3/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@�
.simplified_encoder/BN1/clip_by_value_3/MinimumMinimum!simplified_encoder/BN1/add_13:z:09simplified_encoder/BN1/clip_by_value_3/Minimum/y:output:0*
T0*
_output_shapes
: m
(simplified_encoder/BN1/clip_by_value_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
&simplified_encoder/BN1/clip_by_value_3Maximum2simplified_encoder/BN1/clip_by_value_3/Minimum:z:01simplified_encoder/BN1/clip_by_value_3/y:output:0*
T0*
_output_shapes
: d
simplified_encoder/BN1/mul_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/BN1/mul_14Mul(simplified_encoder/BN1/mul_14/x:output:0*simplified_encoder/BN1/clip_by_value_3:z:0*
T0*
_output_shapes
: �
"simplified_encoder/BN1/SelectV2_11SelectV2!simplified_encoder/BN1/Less_7:z:0!simplified_encoder/BN1/mul_13:z:0!simplified_encoder/BN1/mul_14:z:0*
T0*
_output_shapes
: c
simplified_encoder/BN1/pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
simplified_encoder/BN1/pow_3Pow'simplified_encoder/BN1/pow_3/x:output:0+simplified_encoder/BN1/SelectV2_11:output:0*
T0*
_output_shapes
: �
simplified_encoder/BN1/mul_15Mul!simplified_encoder/BN1/add_11:z:0 simplified_encoder/BN1/pow_3:z:0*
T0*
_output_shapes
: �
'simplified_encoder/BN1/ReadVariableOp_7ReadVariableOp0simplified_encoder_bn1_readvariableop_6_resource*
_output_shapes
: *
dtype0y
simplified_encoder/BN1/Neg_8Neg/simplified_encoder/BN1/ReadVariableOp_7:value:0*
T0*
_output_shapes
: �
simplified_encoder/BN1/add_14AddV2 simplified_encoder/BN1/Neg_8:y:0!simplified_encoder/BN1/mul_15:z:0*
T0*
_output_shapes
: d
simplified_encoder/BN1/mul_16/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/BN1/mul_16Mul(simplified_encoder/BN1/mul_16/x:output:0!simplified_encoder/BN1/add_14:z:0*
T0*
_output_shapes
: }
%simplified_encoder/BN1/StopGradient_6StopGradient!simplified_encoder/BN1/mul_16:z:0*
T0*
_output_shapes
: �
'simplified_encoder/BN1/ReadVariableOp_8ReadVariableOp0simplified_encoder_bn1_readvariableop_6_resource*
_output_shapes
: *
dtype0�
simplified_encoder/BN1/add_15AddV2/simplified_encoder/BN1/ReadVariableOp_8:value:0.simplified_encoder/BN1/StopGradient_6:output:0*
T0*
_output_shapes
: �
,simplified_encoder/BN1/Relu_3/ReadVariableOpReadVariableOp5simplified_encoder_bn1_relu_3_readvariableop_resource*
_output_shapes
: *
dtype0�
simplified_encoder/BN1/Relu_3Relu4simplified_encoder/BN1/Relu_3/ReadVariableOp:value:0*
T0*
_output_shapes
: �
,simplified_encoder/BN1/Relu_4/ReadVariableOpReadVariableOp5simplified_encoder_bn1_relu_3_readvariableop_resource*
_output_shapes
: *
dtype0�
simplified_encoder/BN1/Relu_4Relu4simplified_encoder/BN1/Relu_4/ReadVariableOp:value:0*
T0*
_output_shapes
: d
simplified_encoder/BN1/Less_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
simplified_encoder/BN1/Less_8Less+simplified_encoder/BN1/Relu_4:activations:0(simplified_encoder/BN1/Less_8/y:output:0*
T0*
_output_shapes
: i
$simplified_encoder/BN1/SelectV2_12/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
"simplified_encoder/BN1/SelectV2_12SelectV2!simplified_encoder/BN1/Less_8:z:0-simplified_encoder/BN1/SelectV2_12/t:output:0+simplified_encoder/BN1/Relu_4:activations:0*
T0*
_output_shapes
: d
simplified_encoder/BN1/Less_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
simplified_encoder/BN1/Less_9Less+simplified_encoder/BN1/Relu_4:activations:0(simplified_encoder/BN1/Less_9/y:output:0*
T0*
_output_shapes
: �
8simplified_encoder/BN1/ones_like_7/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: m
(simplified_encoder/BN1/ones_like_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"simplified_encoder/BN1/ones_like_7FillAsimplified_encoder/BN1/ones_like_7/Shape/shape_as_tensor:output:01simplified_encoder/BN1/ones_like_7/Const:output:0*
T0*
_output_shapes
: d
simplified_encoder/BN1/mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
simplified_encoder/BN1/mul_17Mul+simplified_encoder/BN1/ones_like_7:output:0(simplified_encoder/BN1/mul_17/y:output:0*
T0*
_output_shapes
: u
simplified_encoder/BN1/SqrtSqrt+simplified_encoder/BN1/SelectV2_12:output:0*
T0*
_output_shapes
: i
simplified_encoder/BN1/Log_4Logsimplified_encoder/BN1/Sqrt:y:0*
T0*
_output_shapes
: g
"simplified_encoder/BN1/truediv_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?�
 simplified_encoder/BN1/truediv_4RealDiv simplified_encoder/BN1/Log_4:y:0+simplified_encoder/BN1/truediv_4/y:output:0*
T0*
_output_shapes
: n
simplified_encoder/BN1/Neg_9Neg$simplified_encoder/BN1/truediv_4:z:0*
T0*
_output_shapes
: r
simplified_encoder/BN1/Round_4Round$simplified_encoder/BN1/truediv_4:z:0*
T0*
_output_shapes
: �
simplified_encoder/BN1/add_16AddV2 simplified_encoder/BN1/Neg_9:y:0"simplified_encoder/BN1/Round_4:y:0*
T0*
_output_shapes
: }
%simplified_encoder/BN1/StopGradient_7StopGradient!simplified_encoder/BN1/add_16:z:0*
T0*
_output_shapes
: �
simplified_encoder/BN1/add_17AddV2$simplified_encoder/BN1/truediv_4:z:0.simplified_encoder/BN1/StopGradient_7:output:0*
T0*
_output_shapes
: u
0simplified_encoder/BN1/clip_by_value_4/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
.simplified_encoder/BN1/clip_by_value_4/MinimumMinimum!simplified_encoder/BN1/add_17:z:09simplified_encoder/BN1/clip_by_value_4/Minimum/y:output:0*
T0*
_output_shapes
: m
(simplified_encoder/BN1/clip_by_value_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
&simplified_encoder/BN1/clip_by_value_4Maximum2simplified_encoder/BN1/clip_by_value_4/Minimum:z:01simplified_encoder/BN1/clip_by_value_4/y:output:0*
T0*
_output_shapes
: d
simplified_encoder/BN1/mul_18/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
simplified_encoder/BN1/mul_18Mul(simplified_encoder/BN1/mul_18/x:output:0*simplified_encoder/BN1/clip_by_value_4:z:0*
T0*
_output_shapes
: �
"simplified_encoder/BN1/SelectV2_13SelectV2!simplified_encoder/BN1/Less_9:z:0!simplified_encoder/BN1/mul_17:z:0!simplified_encoder/BN1/mul_18:z:0*
T0*
_output_shapes
: �
'simplified_encoder/BN1/ReadVariableOp_9ReadVariableOp5simplified_encoder_bn1_relu_3_readvariableop_resource*
_output_shapes
: *
dtype0z
simplified_encoder/BN1/Neg_10Neg/simplified_encoder/BN1/ReadVariableOp_9:value:0*
T0*
_output_shapes
: m
simplified_encoder/BN1/Relu_5Relu!simplified_encoder/BN1/Neg_10:y:0*
T0*
_output_shapes
: d
simplified_encoder/BN1/mul_19/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
simplified_encoder/BN1/mul_19Mul+simplified_encoder/BN1/Relu_5:activations:0(simplified_encoder/BN1/mul_19/y:output:0*
T0*
_output_shapes
: e
 simplified_encoder/BN1/Less_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
simplified_encoder/BN1/Less_10Less!simplified_encoder/BN1/mul_19:z:0)simplified_encoder/BN1/Less_10/y:output:0*
T0*
_output_shapes
: i
$simplified_encoder/BN1/SelectV2_14/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
"simplified_encoder/BN1/SelectV2_14SelectV2"simplified_encoder/BN1/Less_10:z:0-simplified_encoder/BN1/SelectV2_14/t:output:0!simplified_encoder/BN1/mul_19:z:0*
T0*
_output_shapes
: e
 simplified_encoder/BN1/Less_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
simplified_encoder/BN1/Less_11Less!simplified_encoder/BN1/mul_19:z:0)simplified_encoder/BN1/Less_11/y:output:0*
T0*
_output_shapes
: �
8simplified_encoder/BN1/ones_like_8/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: m
(simplified_encoder/BN1/ones_like_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"simplified_encoder/BN1/ones_like_8FillAsimplified_encoder/BN1/ones_like_8/Shape/shape_as_tensor:output:01simplified_encoder/BN1/ones_like_8/Const:output:0*
T0*
_output_shapes
: d
simplified_encoder/BN1/mul_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
simplified_encoder/BN1/mul_20Mul+simplified_encoder/BN1/ones_like_8:output:0(simplified_encoder/BN1/mul_20/y:output:0*
T0*
_output_shapes
: w
simplified_encoder/BN1/Sqrt_1Sqrt+simplified_encoder/BN1/SelectV2_14:output:0*
T0*
_output_shapes
: k
simplified_encoder/BN1/Log_5Log!simplified_encoder/BN1/Sqrt_1:y:0*
T0*
_output_shapes
: g
"simplified_encoder/BN1/truediv_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?�
 simplified_encoder/BN1/truediv_5RealDiv simplified_encoder/BN1/Log_5:y:0+simplified_encoder/BN1/truediv_5/y:output:0*
T0*
_output_shapes
: o
simplified_encoder/BN1/Neg_11Neg$simplified_encoder/BN1/truediv_5:z:0*
T0*
_output_shapes
: r
simplified_encoder/BN1/Round_5Round$simplified_encoder/BN1/truediv_5:z:0*
T0*
_output_shapes
: �
simplified_encoder/BN1/add_18AddV2!simplified_encoder/BN1/Neg_11:y:0"simplified_encoder/BN1/Round_5:y:0*
T0*
_output_shapes
: }
%simplified_encoder/BN1/StopGradient_8StopGradient!simplified_encoder/BN1/add_18:z:0*
T0*
_output_shapes
: �
simplified_encoder/BN1/add_19AddV2$simplified_encoder/BN1/truediv_5:z:0.simplified_encoder/BN1/StopGradient_8:output:0*
T0*
_output_shapes
: u
0simplified_encoder/BN1/clip_by_value_5/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
.simplified_encoder/BN1/clip_by_value_5/MinimumMinimum!simplified_encoder/BN1/add_19:z:09simplified_encoder/BN1/clip_by_value_5/Minimum/y:output:0*
T0*
_output_shapes
: m
(simplified_encoder/BN1/clip_by_value_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
&simplified_encoder/BN1/clip_by_value_5Maximum2simplified_encoder/BN1/clip_by_value_5/Minimum:z:01simplified_encoder/BN1/clip_by_value_5/y:output:0*
T0*
_output_shapes
: d
simplified_encoder/BN1/mul_21/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
simplified_encoder/BN1/mul_21Mul(simplified_encoder/BN1/mul_21/x:output:0*simplified_encoder/BN1/clip_by_value_5:z:0*
T0*
_output_shapes
: �
"simplified_encoder/BN1/SelectV2_15SelectV2"simplified_encoder/BN1/Less_11:z:0!simplified_encoder/BN1/mul_20:z:0!simplified_encoder/BN1/mul_21:z:0*
T0*
_output_shapes
: �
(simplified_encoder/BN1/ReadVariableOp_10ReadVariableOp5simplified_encoder_bn1_relu_3_readvariableop_resource*
_output_shapes
: *
dtype0l
'simplified_encoder/BN1/GreaterEqual_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
%simplified_encoder/BN1/GreaterEqual_3GreaterEqual0simplified_encoder/BN1/ReadVariableOp_10:value:00simplified_encoder/BN1/GreaterEqual_3/y:output:0*
T0*
_output_shapes
: f
$simplified_encoder/BN1/LogicalOr_1/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z�
"simplified_encoder/BN1/LogicalOr_1	LogicalOr)simplified_encoder/BN1/GreaterEqual_3:z:0-simplified_encoder/BN1/LogicalOr_1/y:output:0*
_output_shapes
: c
simplified_encoder/BN1/pow_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
simplified_encoder/BN1/pow_4Pow'simplified_encoder/BN1/pow_4/x:output:0+simplified_encoder/BN1/SelectV2_13:output:0*
T0*
_output_shapes
: c
simplified_encoder/BN1/pow_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
simplified_encoder/BN1/pow_5Pow'simplified_encoder/BN1/pow_5/x:output:0+simplified_encoder/BN1/SelectV2_15:output:0*
T0*
_output_shapes
: k
simplified_encoder/BN1/Neg_12Neg simplified_encoder/BN1/pow_5:z:0*
T0*
_output_shapes
: �
"simplified_encoder/BN1/SelectV2_16SelectV2&simplified_encoder/BN1/LogicalOr_1:z:0 simplified_encoder/BN1/pow_4:z:0!simplified_encoder/BN1/Neg_12:y:0*
T0*
_output_shapes
: v
simplified_encoder/BN1/Neg_13Neg+simplified_encoder/BN1/Relu_3:activations:0*
T0*
_output_shapes
: �
simplified_encoder/BN1/add_20AddV2!simplified_encoder/BN1/Neg_13:y:0+simplified_encoder/BN1/SelectV2_16:output:0*
T0*
_output_shapes
: d
simplified_encoder/BN1/mul_22/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/BN1/mul_22Mul(simplified_encoder/BN1/mul_22/x:output:0!simplified_encoder/BN1/add_20:z:0*
T0*
_output_shapes
: }
%simplified_encoder/BN1/StopGradient_9StopGradient!simplified_encoder/BN1/mul_22:z:0*
T0*
_output_shapes
: �
simplified_encoder/BN1/add_21AddV2+simplified_encoder/BN1/Relu_3:activations:0.simplified_encoder/BN1/StopGradient_9:output:0*
T0*
_output_shapes
: k
&simplified_encoder/BN1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$simplified_encoder/BN1/batchnorm/addAddV2!simplified_encoder/BN1/add_21:z:0/simplified_encoder/BN1/batchnorm/add/y:output:0*
T0*
_output_shapes
: ~
&simplified_encoder/BN1/batchnorm/RsqrtRsqrt(simplified_encoder/BN1/batchnorm/add:z:0*
T0*
_output_shapes
: �
$simplified_encoder/BN1/batchnorm/mulMul*simplified_encoder/BN1/batchnorm/Rsqrt:y:0 simplified_encoder/BN1/add_5:z:0*
T0*
_output_shapes
: �
&simplified_encoder/BN1/batchnorm/mul_1Mul*simplified_encoder/dense1/BiasAdd:output:0(simplified_encoder/BN1/batchnorm/mul:z:0*
T0*'
_output_shapes
:��������� �
&simplified_encoder/BN1/batchnorm/mul_2Mul!simplified_encoder/BN1/add_15:z:0(simplified_encoder/BN1/batchnorm/mul:z:0*
T0*
_output_shapes
: �
$simplified_encoder/BN1/batchnorm/subSub!simplified_encoder/BN1/add_10:z:0*simplified_encoder/BN1/batchnorm/mul_2:z:0*
T0*
_output_shapes
: �
&simplified_encoder/BN1/batchnorm/add_1AddV2*simplified_encoder/BN1/batchnorm/mul_1:z:0(simplified_encoder/BN1/batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� `
simplified_encoder/relu1/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :`
simplified_encoder/relu1/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :�
simplified_encoder/relu1/PowPow'simplified_encoder/relu1/Pow/x:output:0'simplified_encoder/relu1/Pow/y:output:0*
T0*
_output_shapes
: w
simplified_encoder/relu1/CastCast simplified_encoder/relu1/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: b
 simplified_encoder/relu1/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :b
 simplified_encoder/relu1/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : �
simplified_encoder/relu1/Pow_1Pow)simplified_encoder/relu1/Pow_1/x:output:0)simplified_encoder/relu1/Pow_1/y:output:0*
T0*
_output_shapes
: {
simplified_encoder/relu1/Cast_1Cast"simplified_encoder/relu1/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: c
simplified_encoder/relu1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @c
!simplified_encoder/relu1/Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B : �
simplified_encoder/relu1/Cast_2Cast*simplified_encoder/relu1/Cast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: c
simplified_encoder/relu1/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A�
simplified_encoder/relu1/subSub#simplified_encoder/relu1/Cast_2:y:0'simplified_encoder/relu1/sub/y:output:0*
T0*
_output_shapes
: �
simplified_encoder/relu1/Pow_2Pow'simplified_encoder/relu1/Const:output:0 simplified_encoder/relu1/sub:z:0*
T0*
_output_shapes
: �
simplified_encoder/relu1/sub_1Sub#simplified_encoder/relu1/Cast_1:y:0"simplified_encoder/relu1/Pow_2:z:0*
T0*
_output_shapes
: �
"simplified_encoder/relu1/LessEqual	LessEqual*simplified_encoder/BN1/batchnorm/add_1:z:0"simplified_encoder/relu1/sub_1:z:0*
T0*'
_output_shapes
:��������� �
"simplified_encoder/relu1/LeakyRelu	LeakyRelu*simplified_encoder/BN1/batchnorm/add_1:z:0*'
_output_shapes
:��������� *
alpha%  �:�
(simplified_encoder/relu1/ones_like/ShapeShape*simplified_encoder/BN1/batchnorm/add_1:z:0*
T0*
_output_shapes
::��m
(simplified_encoder/relu1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"simplified_encoder/relu1/ones_likeFill1simplified_encoder/relu1/ones_like/Shape:output:01simplified_encoder/relu1/ones_like/Const:output:0*
T0*'
_output_shapes
:��������� �
simplified_encoder/relu1/sub_2Sub#simplified_encoder/relu1/Cast_1:y:0"simplified_encoder/relu1/Pow_2:z:0*
T0*
_output_shapes
: �
simplified_encoder/relu1/mulMul+simplified_encoder/relu1/ones_like:output:0"simplified_encoder/relu1/sub_2:z:0*
T0*'
_output_shapes
:��������� �
!simplified_encoder/relu1/SelectV2SelectV2&simplified_encoder/relu1/LessEqual:z:00simplified_encoder/relu1/LeakyRelu:activations:0 simplified_encoder/relu1/mul:z:0*
T0*'
_output_shapes
:��������� �
simplified_encoder/relu1/mul_1Mul*simplified_encoder/BN1/batchnorm/add_1:z:0!simplified_encoder/relu1/Cast:y:0*
T0*'
_output_shapes
:��������� �
 simplified_encoder/relu1/truedivRealDiv"simplified_encoder/relu1/mul_1:z:0#simplified_encoder/relu1/Cast_1:y:0*
T0*'
_output_shapes
:��������� {
simplified_encoder/relu1/NegNeg$simplified_encoder/relu1/truediv:z:0*
T0*'
_output_shapes
:��������� 
simplified_encoder/relu1/RoundRound$simplified_encoder/relu1/truediv:z:0*
T0*'
_output_shapes
:��������� �
simplified_encoder/relu1/addAddV2 simplified_encoder/relu1/Neg:y:0"simplified_encoder/relu1/Round:y:0*
T0*'
_output_shapes
:��������� �
%simplified_encoder/relu1/StopGradientStopGradient simplified_encoder/relu1/add:z:0*
T0*'
_output_shapes
:��������� �
simplified_encoder/relu1/add_1AddV2$simplified_encoder/relu1/truediv:z:0.simplified_encoder/relu1/StopGradient:output:0*
T0*'
_output_shapes
:��������� �
"simplified_encoder/relu1/truediv_1RealDiv"simplified_encoder/relu1/add_1:z:0!simplified_encoder/relu1/Cast:y:0*
T0*'
_output_shapes
:��������� i
$simplified_encoder/relu1/truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"simplified_encoder/relu1/truediv_2RealDiv-simplified_encoder/relu1/truediv_2/x:output:0!simplified_encoder/relu1/Cast:y:0*
T0*
_output_shapes
: e
 simplified_encoder/relu1/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/relu1/sub_3Sub)simplified_encoder/relu1/sub_3/x:output:0&simplified_encoder/relu1/truediv_2:z:0*
T0*
_output_shapes
: �
.simplified_encoder/relu1/clip_by_value/MinimumMinimum&simplified_encoder/relu1/truediv_1:z:0"simplified_encoder/relu1/sub_3:z:0*
T0*'
_output_shapes
:��������� m
(simplified_encoder/relu1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
&simplified_encoder/relu1/clip_by_valueMaximum2simplified_encoder/relu1/clip_by_value/Minimum:z:01simplified_encoder/relu1/clip_by_value/y:output:0*
T0*'
_output_shapes
:��������� �
simplified_encoder/relu1/mul_2Mul#simplified_encoder/relu1/Cast_1:y:0*simplified_encoder/relu1/clip_by_value:z:0*
T0*'
_output_shapes
:��������� e
 simplified_encoder/relu1/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �:�
simplified_encoder/relu1/mul_3Mul)simplified_encoder/relu1/mul_3/x:output:0!simplified_encoder/relu1/Cast:y:0*
T0*
_output_shapes
: i
$simplified_encoder/relu1/truediv_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"simplified_encoder/relu1/truediv_3RealDiv-simplified_encoder/relu1/truediv_3/x:output:0"simplified_encoder/relu1/mul_3:z:0*
T0*
_output_shapes
: e
 simplified_encoder/relu1/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �:�
simplified_encoder/relu1/mul_4Mul#simplified_encoder/relu1/Cast_1:y:0)simplified_encoder/relu1/mul_4/y:output:0*
T0*
_output_shapes
: e
 simplified_encoder/relu1/mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �:�
simplified_encoder/relu1/mul_5Mul$simplified_encoder/relu1/truediv:z:0)simplified_encoder/relu1/mul_5/y:output:0*
T0*'
_output_shapes
:��������� {
simplified_encoder/relu1/Neg_1Neg"simplified_encoder/relu1/mul_5:z:0*
T0*'
_output_shapes
:��������� 
 simplified_encoder/relu1/Round_1Round"simplified_encoder/relu1/mul_5:z:0*
T0*'
_output_shapes
:��������� �
simplified_encoder/relu1/add_2AddV2"simplified_encoder/relu1/Neg_1:y:0$simplified_encoder/relu1/Round_1:y:0*
T0*'
_output_shapes
:��������� �
'simplified_encoder/relu1/StopGradient_1StopGradient"simplified_encoder/relu1/add_2:z:0*
T0*'
_output_shapes
:��������� �
simplified_encoder/relu1/add_3AddV2"simplified_encoder/relu1/mul_5:z:00simplified_encoder/relu1/StopGradient_1:output:0*
T0*'
_output_shapes
:��������� �
simplified_encoder/relu1/mul_6Mul"simplified_encoder/relu1/add_3:z:0&simplified_encoder/relu1/truediv_3:z:0*
T0*'
_output_shapes
:��������� w
2simplified_encoder/relu1/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
0simplified_encoder/relu1/clip_by_value_1/MinimumMinimum"simplified_encoder/relu1/mul_6:z:0;simplified_encoder/relu1/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:��������� o
*simplified_encoder/relu1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
(simplified_encoder/relu1/clip_by_value_1Maximum4simplified_encoder/relu1/clip_by_value_1/Minimum:z:03simplified_encoder/relu1/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:��������� �
simplified_encoder/relu1/mul_7Mul"simplified_encoder/relu1/mul_4:z:0,simplified_encoder/relu1/clip_by_value_1:z:0*
T0*'
_output_shapes
:��������� �
simplified_encoder/relu1/add_4AddV2"simplified_encoder/relu1/mul_2:z:0"simplified_encoder/relu1/mul_7:z:0*
T0*'
_output_shapes
:��������� �
simplified_encoder/relu1/Neg_2Neg*simplified_encoder/relu1/SelectV2:output:0*
T0*'
_output_shapes
:��������� �
simplified_encoder/relu1/add_5AddV2"simplified_encoder/relu1/Neg_2:y:0"simplified_encoder/relu1/add_4:z:0*
T0*'
_output_shapes
:��������� e
 simplified_encoder/relu1/mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/relu1/mul_8Mul)simplified_encoder/relu1/mul_8/x:output:0"simplified_encoder/relu1/add_5:z:0*
T0*'
_output_shapes
:��������� �
'simplified_encoder/relu1/StopGradient_2StopGradient"simplified_encoder/relu1/mul_8:z:0*
T0*'
_output_shapes
:��������� �
simplified_encoder/relu1/add_6AddV2*simplified_encoder/relu1/SelectV2:output:00simplified_encoder/relu1/StopGradient_2:output:0*
T0*'
_output_shapes
:��������� a
simplified_encoder/dense2/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :a
simplified_encoder/dense2/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :�
simplified_encoder/dense2/PowPow(simplified_encoder/dense2/Pow/x:output:0(simplified_encoder/dense2/Pow/y:output:0*
T0*
_output_shapes
: y
simplified_encoder/dense2/CastCast!simplified_encoder/dense2/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: �
(simplified_encoder/dense2/ReadVariableOpReadVariableOp1simplified_encoder_dense2_readvariableop_resource*
_output_shapes

: *
dtype0d
simplified_encoder/dense2/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   D�
simplified_encoder/dense2/mulMul0simplified_encoder/dense2/ReadVariableOp:value:0(simplified_encoder/dense2/mul/y:output:0*
T0*
_output_shapes

: �
!simplified_encoder/dense2/truedivRealDiv!simplified_encoder/dense2/mul:z:0"simplified_encoder/dense2/Cast:y:0*
T0*
_output_shapes

: t
simplified_encoder/dense2/NegNeg%simplified_encoder/dense2/truediv:z:0*
T0*
_output_shapes

: x
simplified_encoder/dense2/RoundRound%simplified_encoder/dense2/truediv:z:0*
T0*
_output_shapes

: �
simplified_encoder/dense2/addAddV2!simplified_encoder/dense2/Neg:y:0#simplified_encoder/dense2/Round:y:0*
T0*
_output_shapes

: �
&simplified_encoder/dense2/StopGradientStopGradient!simplified_encoder/dense2/add:z:0*
T0*
_output_shapes

: �
simplified_encoder/dense2/add_1AddV2%simplified_encoder/dense2/truediv:z:0/simplified_encoder/dense2/StopGradient:output:0*
T0*
_output_shapes

: v
1simplified_encoder/dense2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ��C�
/simplified_encoder/dense2/clip_by_value/MinimumMinimum#simplified_encoder/dense2/add_1:z:0:simplified_encoder/dense2/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

: n
)simplified_encoder/dense2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
'simplified_encoder/dense2/clip_by_valueMaximum3simplified_encoder/dense2/clip_by_value/Minimum:z:02simplified_encoder/dense2/clip_by_value/y:output:0*
T0*
_output_shapes

: �
simplified_encoder/dense2/mul_1Mul"simplified_encoder/dense2/Cast:y:0+simplified_encoder/dense2/clip_by_value:z:0*
T0*
_output_shapes

: j
%simplified_encoder/dense2/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   D�
#simplified_encoder/dense2/truediv_1RealDiv#simplified_encoder/dense2/mul_1:z:0.simplified_encoder/dense2/truediv_1/y:output:0*
T0*
_output_shapes

: f
!simplified_encoder/dense2/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/dense2/mul_2Mul*simplified_encoder/dense2/mul_2/x:output:0'simplified_encoder/dense2/truediv_1:z:0*
T0*
_output_shapes

: �
*simplified_encoder/dense2/ReadVariableOp_1ReadVariableOp1simplified_encoder_dense2_readvariableop_resource*
_output_shapes

: *
dtype0�
simplified_encoder/dense2/Neg_1Neg2simplified_encoder/dense2/ReadVariableOp_1:value:0*
T0*
_output_shapes

: �
simplified_encoder/dense2/add_2AddV2#simplified_encoder/dense2/Neg_1:y:0#simplified_encoder/dense2/mul_2:z:0*
T0*
_output_shapes

: f
!simplified_encoder/dense2/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/dense2/mul_3Mul*simplified_encoder/dense2/mul_3/x:output:0#simplified_encoder/dense2/add_2:z:0*
T0*
_output_shapes

: �
(simplified_encoder/dense2/StopGradient_1StopGradient#simplified_encoder/dense2/mul_3:z:0*
T0*
_output_shapes

: �
*simplified_encoder/dense2/ReadVariableOp_2ReadVariableOp1simplified_encoder_dense2_readvariableop_resource*
_output_shapes

: *
dtype0�
simplified_encoder/dense2/add_3AddV22simplified_encoder/dense2/ReadVariableOp_2:value:01simplified_encoder/dense2/StopGradient_1:output:0*
T0*
_output_shapes

: �
 simplified_encoder/dense2/MatMulMatMul"simplified_encoder/relu1/add_6:z:0#simplified_encoder/dense2/add_3:z:0*
T0*'
_output_shapes
:���������c
!simplified_encoder/dense2/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :c
!simplified_encoder/dense2/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
simplified_encoder/dense2/Pow_1Pow*simplified_encoder/dense2/Pow_1/x:output:0*simplified_encoder/dense2/Pow_1/y:output:0*
T0*
_output_shapes
: }
 simplified_encoder/dense2/Cast_1Cast#simplified_encoder/dense2/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
*simplified_encoder/dense2/ReadVariableOp_3ReadVariableOp3simplified_encoder_dense2_readvariableop_3_resource*
_output_shapes
:*
dtype0f
!simplified_encoder/dense2/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   D�
simplified_encoder/dense2/mul_4Mul2simplified_encoder/dense2/ReadVariableOp_3:value:0*simplified_encoder/dense2/mul_4/y:output:0*
T0*
_output_shapes
:�
#simplified_encoder/dense2/truediv_2RealDiv#simplified_encoder/dense2/mul_4:z:0$simplified_encoder/dense2/Cast_1:y:0*
T0*
_output_shapes
:t
simplified_encoder/dense2/Neg_2Neg'simplified_encoder/dense2/truediv_2:z:0*
T0*
_output_shapes
:x
!simplified_encoder/dense2/Round_1Round'simplified_encoder/dense2/truediv_2:z:0*
T0*
_output_shapes
:�
simplified_encoder/dense2/add_4AddV2#simplified_encoder/dense2/Neg_2:y:0%simplified_encoder/dense2/Round_1:y:0*
T0*
_output_shapes
:�
(simplified_encoder/dense2/StopGradient_2StopGradient#simplified_encoder/dense2/add_4:z:0*
T0*
_output_shapes
:�
simplified_encoder/dense2/add_5AddV2'simplified_encoder/dense2/truediv_2:z:01simplified_encoder/dense2/StopGradient_2:output:0*
T0*
_output_shapes
:x
3simplified_encoder/dense2/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ��C�
1simplified_encoder/dense2/clip_by_value_1/MinimumMinimum#simplified_encoder/dense2/add_5:z:0<simplified_encoder/dense2/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:p
+simplified_encoder/dense2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
)simplified_encoder/dense2/clip_by_value_1Maximum5simplified_encoder/dense2/clip_by_value_1/Minimum:z:04simplified_encoder/dense2/clip_by_value_1/y:output:0*
T0*
_output_shapes
:�
simplified_encoder/dense2/mul_5Mul$simplified_encoder/dense2/Cast_1:y:0-simplified_encoder/dense2/clip_by_value_1:z:0*
T0*
_output_shapes
:j
%simplified_encoder/dense2/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   D�
#simplified_encoder/dense2/truediv_3RealDiv#simplified_encoder/dense2/mul_5:z:0.simplified_encoder/dense2/truediv_3/y:output:0*
T0*
_output_shapes
:f
!simplified_encoder/dense2/mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/dense2/mul_6Mul*simplified_encoder/dense2/mul_6/x:output:0'simplified_encoder/dense2/truediv_3:z:0*
T0*
_output_shapes
:�
*simplified_encoder/dense2/ReadVariableOp_4ReadVariableOp3simplified_encoder_dense2_readvariableop_3_resource*
_output_shapes
:*
dtype0
simplified_encoder/dense2/Neg_3Neg2simplified_encoder/dense2/ReadVariableOp_4:value:0*
T0*
_output_shapes
:�
simplified_encoder/dense2/add_6AddV2#simplified_encoder/dense2/Neg_3:y:0#simplified_encoder/dense2/mul_6:z:0*
T0*
_output_shapes
:f
!simplified_encoder/dense2/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/dense2/mul_7Mul*simplified_encoder/dense2/mul_7/x:output:0#simplified_encoder/dense2/add_6:z:0*
T0*
_output_shapes
:�
(simplified_encoder/dense2/StopGradient_3StopGradient#simplified_encoder/dense2/mul_7:z:0*
T0*
_output_shapes
:�
*simplified_encoder/dense2/ReadVariableOp_5ReadVariableOp3simplified_encoder_dense2_readvariableop_3_resource*
_output_shapes
:*
dtype0�
simplified_encoder/dense2/add_7AddV22simplified_encoder/dense2/ReadVariableOp_5:value:01simplified_encoder/dense2/StopGradient_3:output:0*
T0*
_output_shapes
:�
!simplified_encoder/dense2/BiasAddBiasAdd*simplified_encoder/dense2/MatMul:product:0#simplified_encoder/dense2/add_7:z:0*
T0*'
_output_shapes
:����������
%simplified_encoder/BN2/ReadVariableOpReadVariableOp.simplified_encoder_bn2_readvariableop_resource*
_output_shapes
:*
dtype0g
"simplified_encoder/BN2/LessEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   E�
 simplified_encoder/BN2/LessEqual	LessEqual-simplified_encoder/BN2/ReadVariableOp:value:0+simplified_encoder/BN2/LessEqual/y:output:0*
T0*
_output_shapes
:�
*simplified_encoder/BN2/Relu/ReadVariableOpReadVariableOp.simplified_encoder_bn2_readvariableop_resource*
_output_shapes
:*
dtype0|
simplified_encoder/BN2/ReluRelu2simplified_encoder/BN2/Relu/ReadVariableOp:value:0*
T0*
_output_shapes
:�
/simplified_encoder/BN2/ones_like/ReadVariableOpReadVariableOp.simplified_encoder_bn2_readvariableop_resource*
_output_shapes
:*
dtype0�
6simplified_encoder/BN2/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:k
&simplified_encoder/BN2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
 simplified_encoder/BN2/ones_likeFill?simplified_encoder/BN2/ones_like/Shape/shape_as_tensor:output:0/simplified_encoder/BN2/ones_like/Const:output:0*
T0*
_output_shapes
:a
simplified_encoder/BN2/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   E�
simplified_encoder/BN2/mulMul)simplified_encoder/BN2/ones_like:output:0%simplified_encoder/BN2/mul/y:output:0*
T0*
_output_shapes
:�
simplified_encoder/BN2/SelectV2SelectV2$simplified_encoder/BN2/LessEqual:z:0)simplified_encoder/BN2/Relu:activations:0simplified_encoder/BN2/mul:z:0*
T0*
_output_shapes
:�
,simplified_encoder/BN2/Relu_1/ReadVariableOpReadVariableOp.simplified_encoder_bn2_readvariableop_resource*
_output_shapes
:*
dtype0�
simplified_encoder/BN2/Relu_1Relu4simplified_encoder/BN2/Relu_1/ReadVariableOp:value:0*
T0*
_output_shapes
:b
simplified_encoder/BN2/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
simplified_encoder/BN2/LessLess+simplified_encoder/BN2/Relu_1:activations:0&simplified_encoder/BN2/Less/y:output:0*
T0*
_output_shapes
:h
#simplified_encoder/BN2/SelectV2_1/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
!simplified_encoder/BN2/SelectV2_1SelectV2simplified_encoder/BN2/Less:z:0,simplified_encoder/BN2/SelectV2_1/t:output:0+simplified_encoder/BN2/Relu_1:activations:0*
T0*
_output_shapes
:j
%simplified_encoder/BN2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   E�
#simplified_encoder/BN2/GreaterEqualGreaterEqual*simplified_encoder/BN2/SelectV2_1:output:0.simplified_encoder/BN2/GreaterEqual/y:output:0*
T0*
_output_shapes
:�
8simplified_encoder/BN2/ones_like_1/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:m
(simplified_encoder/BN2/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"simplified_encoder/BN2/ones_like_1FillAsimplified_encoder/BN2/ones_like_1/Shape/shape_as_tensor:output:01simplified_encoder/BN2/ones_like_1/Const:output:0*
T0*
_output_shapes
:c
simplified_encoder/BN2/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   E�
simplified_encoder/BN2/mul_1Mul+simplified_encoder/BN2/ones_like_1:output:0'simplified_encoder/BN2/mul_1/y:output:0*
T0*
_output_shapes
:�
!simplified_encoder/BN2/SelectV2_2SelectV2'simplified_encoder/BN2/GreaterEqual:z:0 simplified_encoder/BN2/mul_1:z:0*simplified_encoder/BN2/SelectV2_1:output:0*
T0*
_output_shapes
:d
simplified_encoder/BN2/Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
simplified_encoder/BN2/Less_1Less+simplified_encoder/BN2/Relu_1:activations:0(simplified_encoder/BN2/Less_1/y:output:0*
T0*
_output_shapes
:�
8simplified_encoder/BN2/ones_like_2/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:m
(simplified_encoder/BN2/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"simplified_encoder/BN2/ones_like_2FillAsimplified_encoder/BN2/ones_like_2/Shape/shape_as_tensor:output:01simplified_encoder/BN2/ones_like_2/Const:output:0*
T0*
_output_shapes
:c
simplified_encoder/BN2/mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
simplified_encoder/BN2/mul_2Mul+simplified_encoder/BN2/ones_like_2:output:0'simplified_encoder/BN2/mul_2/y:output:0*
T0*
_output_shapes
:r
simplified_encoder/BN2/LogLog*simplified_encoder/BN2/SelectV2_2:output:0*
T0*
_output_shapes
:e
 simplified_encoder/BN2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?�
simplified_encoder/BN2/truedivRealDivsimplified_encoder/BN2/Log:y:0)simplified_encoder/BN2/truediv/y:output:0*
T0*
_output_shapes
:j
simplified_encoder/BN2/NegNeg"simplified_encoder/BN2/truediv:z:0*
T0*
_output_shapes
:n
simplified_encoder/BN2/RoundRound"simplified_encoder/BN2/truediv:z:0*
T0*
_output_shapes
:�
simplified_encoder/BN2/addAddV2simplified_encoder/BN2/Neg:y:0 simplified_encoder/BN2/Round:y:0*
T0*
_output_shapes
:x
#simplified_encoder/BN2/StopGradientStopGradientsimplified_encoder/BN2/add:z:0*
T0*
_output_shapes
:�
simplified_encoder/BN2/add_1AddV2"simplified_encoder/BN2/truediv:z:0,simplified_encoder/BN2/StopGradient:output:0*
T0*
_output_shapes
:s
.simplified_encoder/BN2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
,simplified_encoder/BN2/clip_by_value/MinimumMinimum simplified_encoder/BN2/add_1:z:07simplified_encoder/BN2/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:k
&simplified_encoder/BN2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
$simplified_encoder/BN2/clip_by_valueMaximum0simplified_encoder/BN2/clip_by_value/Minimum:z:0/simplified_encoder/BN2/clip_by_value/y:output:0*
T0*
_output_shapes
:c
simplified_encoder/BN2/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/BN2/mul_3Mul'simplified_encoder/BN2/mul_3/x:output:0(simplified_encoder/BN2/clip_by_value:z:0*
T0*
_output_shapes
:�
!simplified_encoder/BN2/SelectV2_3SelectV2!simplified_encoder/BN2/Less_1:z:0 simplified_encoder/BN2/mul_2:z:0 simplified_encoder/BN2/mul_3:z:0*
T0*
_output_shapes
:�
'simplified_encoder/BN2/ReadVariableOp_1ReadVariableOp.simplified_encoder_bn2_readvariableop_resource*
_output_shapes
:*
dtype0y
simplified_encoder/BN2/Neg_1Neg/simplified_encoder/BN2/ReadVariableOp_1:value:0*
T0*
_output_shapes
:l
simplified_encoder/BN2/Relu_2Relu simplified_encoder/BN2/Neg_1:y:0*
T0*
_output_shapes
:c
simplified_encoder/BN2/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
simplified_encoder/BN2/mul_4Mul+simplified_encoder/BN2/Relu_2:activations:0'simplified_encoder/BN2/mul_4/y:output:0*
T0*
_output_shapes
:d
simplified_encoder/BN2/Less_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
simplified_encoder/BN2/Less_2Less simplified_encoder/BN2/mul_4:z:0(simplified_encoder/BN2/Less_2/y:output:0*
T0*
_output_shapes
:h
#simplified_encoder/BN2/SelectV2_4/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
!simplified_encoder/BN2/SelectV2_4SelectV2!simplified_encoder/BN2/Less_2:z:0,simplified_encoder/BN2/SelectV2_4/t:output:0 simplified_encoder/BN2/mul_4:z:0*
T0*
_output_shapes
:l
'simplified_encoder/BN2/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   E�
%simplified_encoder/BN2/GreaterEqual_1GreaterEqual*simplified_encoder/BN2/SelectV2_4:output:00simplified_encoder/BN2/GreaterEqual_1/y:output:0*
T0*
_output_shapes
:�
8simplified_encoder/BN2/ones_like_3/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:m
(simplified_encoder/BN2/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"simplified_encoder/BN2/ones_like_3FillAsimplified_encoder/BN2/ones_like_3/Shape/shape_as_tensor:output:01simplified_encoder/BN2/ones_like_3/Const:output:0*
T0*
_output_shapes
:c
simplified_encoder/BN2/mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *   E�
simplified_encoder/BN2/mul_5Mul+simplified_encoder/BN2/ones_like_3:output:0'simplified_encoder/BN2/mul_5/y:output:0*
T0*
_output_shapes
:�
!simplified_encoder/BN2/SelectV2_5SelectV2)simplified_encoder/BN2/GreaterEqual_1:z:0 simplified_encoder/BN2/mul_5:z:0*simplified_encoder/BN2/SelectV2_4:output:0*
T0*
_output_shapes
:d
simplified_encoder/BN2/Less_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
simplified_encoder/BN2/Less_3Less simplified_encoder/BN2/mul_4:z:0(simplified_encoder/BN2/Less_3/y:output:0*
T0*
_output_shapes
:�
8simplified_encoder/BN2/ones_like_4/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:m
(simplified_encoder/BN2/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"simplified_encoder/BN2/ones_like_4FillAsimplified_encoder/BN2/ones_like_4/Shape/shape_as_tensor:output:01simplified_encoder/BN2/ones_like_4/Const:output:0*
T0*
_output_shapes
:c
simplified_encoder/BN2/mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
simplified_encoder/BN2/mul_6Mul+simplified_encoder/BN2/ones_like_4:output:0'simplified_encoder/BN2/mul_6/y:output:0*
T0*
_output_shapes
:t
simplified_encoder/BN2/Log_1Log*simplified_encoder/BN2/SelectV2_5:output:0*
T0*
_output_shapes
:g
"simplified_encoder/BN2/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?�
 simplified_encoder/BN2/truediv_1RealDiv simplified_encoder/BN2/Log_1:y:0+simplified_encoder/BN2/truediv_1/y:output:0*
T0*
_output_shapes
:n
simplified_encoder/BN2/Neg_2Neg$simplified_encoder/BN2/truediv_1:z:0*
T0*
_output_shapes
:r
simplified_encoder/BN2/Round_1Round$simplified_encoder/BN2/truediv_1:z:0*
T0*
_output_shapes
:�
simplified_encoder/BN2/add_2AddV2 simplified_encoder/BN2/Neg_2:y:0"simplified_encoder/BN2/Round_1:y:0*
T0*
_output_shapes
:|
%simplified_encoder/BN2/StopGradient_1StopGradient simplified_encoder/BN2/add_2:z:0*
T0*
_output_shapes
:�
simplified_encoder/BN2/add_3AddV2$simplified_encoder/BN2/truediv_1:z:0.simplified_encoder/BN2/StopGradient_1:output:0*
T0*
_output_shapes
:u
0simplified_encoder/BN2/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
.simplified_encoder/BN2/clip_by_value_1/MinimumMinimum simplified_encoder/BN2/add_3:z:09simplified_encoder/BN2/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:m
(simplified_encoder/BN2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
&simplified_encoder/BN2/clip_by_value_1Maximum2simplified_encoder/BN2/clip_by_value_1/Minimum:z:01simplified_encoder/BN2/clip_by_value_1/y:output:0*
T0*
_output_shapes
:c
simplified_encoder/BN2/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/BN2/mul_7Mul'simplified_encoder/BN2/mul_7/x:output:0*simplified_encoder/BN2/clip_by_value_1:z:0*
T0*
_output_shapes
:�
!simplified_encoder/BN2/SelectV2_6SelectV2!simplified_encoder/BN2/Less_3:z:0 simplified_encoder/BN2/mul_6:z:0 simplified_encoder/BN2/mul_7:z:0*
T0*
_output_shapes
:�
'simplified_encoder/BN2/ReadVariableOp_2ReadVariableOp.simplified_encoder_bn2_readvariableop_resource*
_output_shapes
:*
dtype0l
'simplified_encoder/BN2/GreaterEqual_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
%simplified_encoder/BN2/GreaterEqual_2GreaterEqual/simplified_encoder/BN2/ReadVariableOp_2:value:00simplified_encoder/BN2/GreaterEqual_2/y:output:0*
T0*
_output_shapes
:d
"simplified_encoder/BN2/LogicalOr/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z�
 simplified_encoder/BN2/LogicalOr	LogicalOr)simplified_encoder/BN2/GreaterEqual_2:z:0+simplified_encoder/BN2/LogicalOr/y:output:0*
_output_shapes
:a
simplified_encoder/BN2/pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
simplified_encoder/BN2/powPow%simplified_encoder/BN2/pow/x:output:0*simplified_encoder/BN2/SelectV2_3:output:0*
T0*
_output_shapes
:c
simplified_encoder/BN2/pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
simplified_encoder/BN2/pow_1Pow'simplified_encoder/BN2/pow_1/x:output:0*simplified_encoder/BN2/SelectV2_6:output:0*
T0*
_output_shapes
:j
simplified_encoder/BN2/Neg_3Neg simplified_encoder/BN2/pow_1:z:0*
T0*
_output_shapes
:�
!simplified_encoder/BN2/SelectV2_7SelectV2$simplified_encoder/BN2/LogicalOr:z:0simplified_encoder/BN2/pow:z:0 simplified_encoder/BN2/Neg_3:y:0*
T0*
_output_shapes
:r
simplified_encoder/BN2/Neg_4Neg(simplified_encoder/BN2/SelectV2:output:0*
T0*
_output_shapes
:�
simplified_encoder/BN2/add_4AddV2 simplified_encoder/BN2/Neg_4:y:0*simplified_encoder/BN2/SelectV2_7:output:0*
T0*
_output_shapes
:c
simplified_encoder/BN2/mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/BN2/mul_8Mul'simplified_encoder/BN2/mul_8/x:output:0 simplified_encoder/BN2/add_4:z:0*
T0*
_output_shapes
:|
%simplified_encoder/BN2/StopGradient_2StopGradient simplified_encoder/BN2/mul_8:z:0*
T0*
_output_shapes
:�
simplified_encoder/BN2/add_5AddV2(simplified_encoder/BN2/SelectV2:output:0.simplified_encoder/BN2/StopGradient_2:output:0*
T0*
_output_shapes
:�
'simplified_encoder/BN2/ReadVariableOp_3ReadVariableOp0simplified_encoder_bn2_readvariableop_3_resource*
_output_shapes
:*
dtype0y
simplified_encoder/BN2/SignSign/simplified_encoder/BN2/ReadVariableOp_3:value:0*
T0*
_output_shapes
:g
simplified_encoder/BN2/AbsAbssimplified_encoder/BN2/Sign:y:0*
T0*
_output_shapes
:a
simplified_encoder/BN2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/BN2/subSub%simplified_encoder/BN2/sub/x:output:0simplified_encoder/BN2/Abs:y:0*
T0*
_output_shapes
:�
simplified_encoder/BN2/add_6AddV2simplified_encoder/BN2/Sign:y:0simplified_encoder/BN2/sub:z:0*
T0*
_output_shapes
:�
+simplified_encoder/BN2/Abs_1/ReadVariableOpReadVariableOp0simplified_encoder_bn2_readvariableop_3_resource*
_output_shapes
:*
dtype0}
simplified_encoder/BN2/Abs_1Abs3simplified_encoder/BN2/Abs_1/ReadVariableOp:value:0*
T0*
_output_shapes
:d
simplified_encoder/BN2/Less_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
simplified_encoder/BN2/Less_4Less simplified_encoder/BN2/Abs_1:y:0(simplified_encoder/BN2/Less_4/y:output:0*
T0*
_output_shapes
:h
#simplified_encoder/BN2/SelectV2_8/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
!simplified_encoder/BN2/SelectV2_8SelectV2!simplified_encoder/BN2/Less_4:z:0,simplified_encoder/BN2/SelectV2_8/t:output:0 simplified_encoder/BN2/Abs_1:y:0*
T0*
_output_shapes
:d
simplified_encoder/BN2/Less_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
simplified_encoder/BN2/Less_5Less simplified_encoder/BN2/Abs_1:y:0(simplified_encoder/BN2/Less_5/y:output:0*
T0*
_output_shapes
:�
8simplified_encoder/BN2/ones_like_5/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:m
(simplified_encoder/BN2/ones_like_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"simplified_encoder/BN2/ones_like_5FillAsimplified_encoder/BN2/ones_like_5/Shape/shape_as_tensor:output:01simplified_encoder/BN2/ones_like_5/Const:output:0*
T0*
_output_shapes
:c
simplified_encoder/BN2/mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
simplified_encoder/BN2/mul_9Mul+simplified_encoder/BN2/ones_like_5:output:0'simplified_encoder/BN2/mul_9/y:output:0*
T0*
_output_shapes
:t
simplified_encoder/BN2/Log_2Log*simplified_encoder/BN2/SelectV2_8:output:0*
T0*
_output_shapes
:g
"simplified_encoder/BN2/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?�
 simplified_encoder/BN2/truediv_2RealDiv simplified_encoder/BN2/Log_2:y:0+simplified_encoder/BN2/truediv_2/y:output:0*
T0*
_output_shapes
:n
simplified_encoder/BN2/Neg_5Neg$simplified_encoder/BN2/truediv_2:z:0*
T0*
_output_shapes
:r
simplified_encoder/BN2/Round_2Round$simplified_encoder/BN2/truediv_2:z:0*
T0*
_output_shapes
:�
simplified_encoder/BN2/add_7AddV2 simplified_encoder/BN2/Neg_5:y:0"simplified_encoder/BN2/Round_2:y:0*
T0*
_output_shapes
:|
%simplified_encoder/BN2/StopGradient_3StopGradient simplified_encoder/BN2/add_7:z:0*
T0*
_output_shapes
:�
simplified_encoder/BN2/add_8AddV2$simplified_encoder/BN2/truediv_2:z:0.simplified_encoder/BN2/StopGradient_3:output:0*
T0*
_output_shapes
:u
0simplified_encoder/BN2/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@�
.simplified_encoder/BN2/clip_by_value_2/MinimumMinimum simplified_encoder/BN2/add_8:z:09simplified_encoder/BN2/clip_by_value_2/Minimum/y:output:0*
T0*
_output_shapes
:m
(simplified_encoder/BN2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
&simplified_encoder/BN2/clip_by_value_2Maximum2simplified_encoder/BN2/clip_by_value_2/Minimum:z:01simplified_encoder/BN2/clip_by_value_2/y:output:0*
T0*
_output_shapes
:d
simplified_encoder/BN2/mul_10/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/BN2/mul_10Mul(simplified_encoder/BN2/mul_10/x:output:0*simplified_encoder/BN2/clip_by_value_2:z:0*
T0*
_output_shapes
:�
!simplified_encoder/BN2/SelectV2_9SelectV2!simplified_encoder/BN2/Less_5:z:0 simplified_encoder/BN2/mul_9:z:0!simplified_encoder/BN2/mul_10:z:0*
T0*
_output_shapes
:c
simplified_encoder/BN2/pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
simplified_encoder/BN2/pow_2Pow'simplified_encoder/BN2/pow_2/x:output:0*simplified_encoder/BN2/SelectV2_9:output:0*
T0*
_output_shapes
:�
simplified_encoder/BN2/mul_11Mul simplified_encoder/BN2/add_6:z:0 simplified_encoder/BN2/pow_2:z:0*
T0*
_output_shapes
:�
'simplified_encoder/BN2/ReadVariableOp_4ReadVariableOp0simplified_encoder_bn2_readvariableop_3_resource*
_output_shapes
:*
dtype0y
simplified_encoder/BN2/Neg_6Neg/simplified_encoder/BN2/ReadVariableOp_4:value:0*
T0*
_output_shapes
:�
simplified_encoder/BN2/add_9AddV2 simplified_encoder/BN2/Neg_6:y:0!simplified_encoder/BN2/mul_11:z:0*
T0*
_output_shapes
:d
simplified_encoder/BN2/mul_12/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/BN2/mul_12Mul(simplified_encoder/BN2/mul_12/x:output:0 simplified_encoder/BN2/add_9:z:0*
T0*
_output_shapes
:}
%simplified_encoder/BN2/StopGradient_4StopGradient!simplified_encoder/BN2/mul_12:z:0*
T0*
_output_shapes
:�
'simplified_encoder/BN2/ReadVariableOp_5ReadVariableOp0simplified_encoder_bn2_readvariableop_3_resource*
_output_shapes
:*
dtype0�
simplified_encoder/BN2/add_10AddV2/simplified_encoder/BN2/ReadVariableOp_5:value:0.simplified_encoder/BN2/StopGradient_4:output:0*
T0*
_output_shapes
:�
'simplified_encoder/BN2/ReadVariableOp_6ReadVariableOp0simplified_encoder_bn2_readvariableop_6_resource*
_output_shapes
:*
dtype0{
simplified_encoder/BN2/Sign_1Sign/simplified_encoder/BN2/ReadVariableOp_6:value:0*
T0*
_output_shapes
:k
simplified_encoder/BN2/Abs_2Abs!simplified_encoder/BN2/Sign_1:y:0*
T0*
_output_shapes
:c
simplified_encoder/BN2/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/BN2/sub_1Sub'simplified_encoder/BN2/sub_1/x:output:0 simplified_encoder/BN2/Abs_2:y:0*
T0*
_output_shapes
:�
simplified_encoder/BN2/add_11AddV2!simplified_encoder/BN2/Sign_1:y:0 simplified_encoder/BN2/sub_1:z:0*
T0*
_output_shapes
:�
+simplified_encoder/BN2/Abs_3/ReadVariableOpReadVariableOp0simplified_encoder_bn2_readvariableop_6_resource*
_output_shapes
:*
dtype0}
simplified_encoder/BN2/Abs_3Abs3simplified_encoder/BN2/Abs_3/ReadVariableOp:value:0*
T0*
_output_shapes
:d
simplified_encoder/BN2/Less_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
simplified_encoder/BN2/Less_6Less simplified_encoder/BN2/Abs_3:y:0(simplified_encoder/BN2/Less_6/y:output:0*
T0*
_output_shapes
:i
$simplified_encoder/BN2/SelectV2_10/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
"simplified_encoder/BN2/SelectV2_10SelectV2!simplified_encoder/BN2/Less_6:z:0-simplified_encoder/BN2/SelectV2_10/t:output:0 simplified_encoder/BN2/Abs_3:y:0*
T0*
_output_shapes
:d
simplified_encoder/BN2/Less_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
simplified_encoder/BN2/Less_7Less simplified_encoder/BN2/Abs_3:y:0(simplified_encoder/BN2/Less_7/y:output:0*
T0*
_output_shapes
:�
8simplified_encoder/BN2/ones_like_6/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:m
(simplified_encoder/BN2/ones_like_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"simplified_encoder/BN2/ones_like_6FillAsimplified_encoder/BN2/ones_like_6/Shape/shape_as_tensor:output:01simplified_encoder/BN2/ones_like_6/Const:output:0*
T0*
_output_shapes
:d
simplified_encoder/BN2/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
simplified_encoder/BN2/mul_13Mul+simplified_encoder/BN2/ones_like_6:output:0(simplified_encoder/BN2/mul_13/y:output:0*
T0*
_output_shapes
:u
simplified_encoder/BN2/Log_3Log+simplified_encoder/BN2/SelectV2_10:output:0*
T0*
_output_shapes
:g
"simplified_encoder/BN2/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?�
 simplified_encoder/BN2/truediv_3RealDiv simplified_encoder/BN2/Log_3:y:0+simplified_encoder/BN2/truediv_3/y:output:0*
T0*
_output_shapes
:n
simplified_encoder/BN2/Neg_7Neg$simplified_encoder/BN2/truediv_3:z:0*
T0*
_output_shapes
:r
simplified_encoder/BN2/Round_3Round$simplified_encoder/BN2/truediv_3:z:0*
T0*
_output_shapes
:�
simplified_encoder/BN2/add_12AddV2 simplified_encoder/BN2/Neg_7:y:0"simplified_encoder/BN2/Round_3:y:0*
T0*
_output_shapes
:}
%simplified_encoder/BN2/StopGradient_5StopGradient!simplified_encoder/BN2/add_12:z:0*
T0*
_output_shapes
:�
simplified_encoder/BN2/add_13AddV2$simplified_encoder/BN2/truediv_3:z:0.simplified_encoder/BN2/StopGradient_5:output:0*
T0*
_output_shapes
:u
0simplified_encoder/BN2/clip_by_value_3/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@�
.simplified_encoder/BN2/clip_by_value_3/MinimumMinimum!simplified_encoder/BN2/add_13:z:09simplified_encoder/BN2/clip_by_value_3/Minimum/y:output:0*
T0*
_output_shapes
:m
(simplified_encoder/BN2/clip_by_value_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
&simplified_encoder/BN2/clip_by_value_3Maximum2simplified_encoder/BN2/clip_by_value_3/Minimum:z:01simplified_encoder/BN2/clip_by_value_3/y:output:0*
T0*
_output_shapes
:d
simplified_encoder/BN2/mul_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/BN2/mul_14Mul(simplified_encoder/BN2/mul_14/x:output:0*simplified_encoder/BN2/clip_by_value_3:z:0*
T0*
_output_shapes
:�
"simplified_encoder/BN2/SelectV2_11SelectV2!simplified_encoder/BN2/Less_7:z:0!simplified_encoder/BN2/mul_13:z:0!simplified_encoder/BN2/mul_14:z:0*
T0*
_output_shapes
:c
simplified_encoder/BN2/pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
simplified_encoder/BN2/pow_3Pow'simplified_encoder/BN2/pow_3/x:output:0+simplified_encoder/BN2/SelectV2_11:output:0*
T0*
_output_shapes
:�
simplified_encoder/BN2/mul_15Mul!simplified_encoder/BN2/add_11:z:0 simplified_encoder/BN2/pow_3:z:0*
T0*
_output_shapes
:�
'simplified_encoder/BN2/ReadVariableOp_7ReadVariableOp0simplified_encoder_bn2_readvariableop_6_resource*
_output_shapes
:*
dtype0y
simplified_encoder/BN2/Neg_8Neg/simplified_encoder/BN2/ReadVariableOp_7:value:0*
T0*
_output_shapes
:�
simplified_encoder/BN2/add_14AddV2 simplified_encoder/BN2/Neg_8:y:0!simplified_encoder/BN2/mul_15:z:0*
T0*
_output_shapes
:d
simplified_encoder/BN2/mul_16/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/BN2/mul_16Mul(simplified_encoder/BN2/mul_16/x:output:0!simplified_encoder/BN2/add_14:z:0*
T0*
_output_shapes
:}
%simplified_encoder/BN2/StopGradient_6StopGradient!simplified_encoder/BN2/mul_16:z:0*
T0*
_output_shapes
:�
'simplified_encoder/BN2/ReadVariableOp_8ReadVariableOp0simplified_encoder_bn2_readvariableop_6_resource*
_output_shapes
:*
dtype0�
simplified_encoder/BN2/add_15AddV2/simplified_encoder/BN2/ReadVariableOp_8:value:0.simplified_encoder/BN2/StopGradient_6:output:0*
T0*
_output_shapes
:�
,simplified_encoder/BN2/Relu_3/ReadVariableOpReadVariableOp5simplified_encoder_bn2_relu_3_readvariableop_resource*
_output_shapes
:*
dtype0�
simplified_encoder/BN2/Relu_3Relu4simplified_encoder/BN2/Relu_3/ReadVariableOp:value:0*
T0*
_output_shapes
:�
,simplified_encoder/BN2/Relu_4/ReadVariableOpReadVariableOp5simplified_encoder_bn2_relu_3_readvariableop_resource*
_output_shapes
:*
dtype0�
simplified_encoder/BN2/Relu_4Relu4simplified_encoder/BN2/Relu_4/ReadVariableOp:value:0*
T0*
_output_shapes
:d
simplified_encoder/BN2/Less_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
simplified_encoder/BN2/Less_8Less+simplified_encoder/BN2/Relu_4:activations:0(simplified_encoder/BN2/Less_8/y:output:0*
T0*
_output_shapes
:i
$simplified_encoder/BN2/SelectV2_12/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
"simplified_encoder/BN2/SelectV2_12SelectV2!simplified_encoder/BN2/Less_8:z:0-simplified_encoder/BN2/SelectV2_12/t:output:0+simplified_encoder/BN2/Relu_4:activations:0*
T0*
_output_shapes
:d
simplified_encoder/BN2/Less_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
simplified_encoder/BN2/Less_9Less+simplified_encoder/BN2/Relu_4:activations:0(simplified_encoder/BN2/Less_9/y:output:0*
T0*
_output_shapes
:�
8simplified_encoder/BN2/ones_like_7/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:m
(simplified_encoder/BN2/ones_like_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"simplified_encoder/BN2/ones_like_7FillAsimplified_encoder/BN2/ones_like_7/Shape/shape_as_tensor:output:01simplified_encoder/BN2/ones_like_7/Const:output:0*
T0*
_output_shapes
:d
simplified_encoder/BN2/mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
simplified_encoder/BN2/mul_17Mul+simplified_encoder/BN2/ones_like_7:output:0(simplified_encoder/BN2/mul_17/y:output:0*
T0*
_output_shapes
:u
simplified_encoder/BN2/SqrtSqrt+simplified_encoder/BN2/SelectV2_12:output:0*
T0*
_output_shapes
:i
simplified_encoder/BN2/Log_4Logsimplified_encoder/BN2/Sqrt:y:0*
T0*
_output_shapes
:g
"simplified_encoder/BN2/truediv_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?�
 simplified_encoder/BN2/truediv_4RealDiv simplified_encoder/BN2/Log_4:y:0+simplified_encoder/BN2/truediv_4/y:output:0*
T0*
_output_shapes
:n
simplified_encoder/BN2/Neg_9Neg$simplified_encoder/BN2/truediv_4:z:0*
T0*
_output_shapes
:r
simplified_encoder/BN2/Round_4Round$simplified_encoder/BN2/truediv_4:z:0*
T0*
_output_shapes
:�
simplified_encoder/BN2/add_16AddV2 simplified_encoder/BN2/Neg_9:y:0"simplified_encoder/BN2/Round_4:y:0*
T0*
_output_shapes
:}
%simplified_encoder/BN2/StopGradient_7StopGradient!simplified_encoder/BN2/add_16:z:0*
T0*
_output_shapes
:�
simplified_encoder/BN2/add_17AddV2$simplified_encoder/BN2/truediv_4:z:0.simplified_encoder/BN2/StopGradient_7:output:0*
T0*
_output_shapes
:u
0simplified_encoder/BN2/clip_by_value_4/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
.simplified_encoder/BN2/clip_by_value_4/MinimumMinimum!simplified_encoder/BN2/add_17:z:09simplified_encoder/BN2/clip_by_value_4/Minimum/y:output:0*
T0*
_output_shapes
:m
(simplified_encoder/BN2/clip_by_value_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
&simplified_encoder/BN2/clip_by_value_4Maximum2simplified_encoder/BN2/clip_by_value_4/Minimum:z:01simplified_encoder/BN2/clip_by_value_4/y:output:0*
T0*
_output_shapes
:d
simplified_encoder/BN2/mul_18/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
simplified_encoder/BN2/mul_18Mul(simplified_encoder/BN2/mul_18/x:output:0*simplified_encoder/BN2/clip_by_value_4:z:0*
T0*
_output_shapes
:�
"simplified_encoder/BN2/SelectV2_13SelectV2!simplified_encoder/BN2/Less_9:z:0!simplified_encoder/BN2/mul_17:z:0!simplified_encoder/BN2/mul_18:z:0*
T0*
_output_shapes
:�
'simplified_encoder/BN2/ReadVariableOp_9ReadVariableOp5simplified_encoder_bn2_relu_3_readvariableop_resource*
_output_shapes
:*
dtype0z
simplified_encoder/BN2/Neg_10Neg/simplified_encoder/BN2/ReadVariableOp_9:value:0*
T0*
_output_shapes
:m
simplified_encoder/BN2/Relu_5Relu!simplified_encoder/BN2/Neg_10:y:0*
T0*
_output_shapes
:d
simplified_encoder/BN2/mul_19/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
simplified_encoder/BN2/mul_19Mul+simplified_encoder/BN2/Relu_5:activations:0(simplified_encoder/BN2/mul_19/y:output:0*
T0*
_output_shapes
:e
 simplified_encoder/BN2/Less_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
simplified_encoder/BN2/Less_10Less!simplified_encoder/BN2/mul_19:z:0)simplified_encoder/BN2/Less_10/y:output:0*
T0*
_output_shapes
:i
$simplified_encoder/BN2/SelectV2_14/tConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
"simplified_encoder/BN2/SelectV2_14SelectV2"simplified_encoder/BN2/Less_10:z:0-simplified_encoder/BN2/SelectV2_14/t:output:0!simplified_encoder/BN2/mul_19:z:0*
T0*
_output_shapes
:e
 simplified_encoder/BN2/Less_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
simplified_encoder/BN2/Less_11Less!simplified_encoder/BN2/mul_19:z:0)simplified_encoder/BN2/Less_11/y:output:0*
T0*
_output_shapes
:�
8simplified_encoder/BN2/ones_like_8/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:m
(simplified_encoder/BN2/ones_like_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"simplified_encoder/BN2/ones_like_8FillAsimplified_encoder/BN2/ones_like_8/Shape/shape_as_tensor:output:01simplified_encoder/BN2/ones_like_8/Const:output:0*
T0*
_output_shapes
:d
simplified_encoder/BN2/mul_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
simplified_encoder/BN2/mul_20Mul+simplified_encoder/BN2/ones_like_8:output:0(simplified_encoder/BN2/mul_20/y:output:0*
T0*
_output_shapes
:w
simplified_encoder/BN2/Sqrt_1Sqrt+simplified_encoder/BN2/SelectV2_14:output:0*
T0*
_output_shapes
:k
simplified_encoder/BN2/Log_5Log!simplified_encoder/BN2/Sqrt_1:y:0*
T0*
_output_shapes
:g
"simplified_encoder/BN2/truediv_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?�
 simplified_encoder/BN2/truediv_5RealDiv simplified_encoder/BN2/Log_5:y:0+simplified_encoder/BN2/truediv_5/y:output:0*
T0*
_output_shapes
:o
simplified_encoder/BN2/Neg_11Neg$simplified_encoder/BN2/truediv_5:z:0*
T0*
_output_shapes
:r
simplified_encoder/BN2/Round_5Round$simplified_encoder/BN2/truediv_5:z:0*
T0*
_output_shapes
:�
simplified_encoder/BN2/add_18AddV2!simplified_encoder/BN2/Neg_11:y:0"simplified_encoder/BN2/Round_5:y:0*
T0*
_output_shapes
:}
%simplified_encoder/BN2/StopGradient_8StopGradient!simplified_encoder/BN2/add_18:z:0*
T0*
_output_shapes
:�
simplified_encoder/BN2/add_19AddV2$simplified_encoder/BN2/truediv_5:z:0.simplified_encoder/BN2/StopGradient_8:output:0*
T0*
_output_shapes
:u
0simplified_encoder/BN2/clip_by_value_5/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
.simplified_encoder/BN2/clip_by_value_5/MinimumMinimum!simplified_encoder/BN2/add_19:z:09simplified_encoder/BN2/clip_by_value_5/Minimum/y:output:0*
T0*
_output_shapes
:m
(simplified_encoder/BN2/clip_by_value_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
&simplified_encoder/BN2/clip_by_value_5Maximum2simplified_encoder/BN2/clip_by_value_5/Minimum:z:01simplified_encoder/BN2/clip_by_value_5/y:output:0*
T0*
_output_shapes
:d
simplified_encoder/BN2/mul_21/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
simplified_encoder/BN2/mul_21Mul(simplified_encoder/BN2/mul_21/x:output:0*simplified_encoder/BN2/clip_by_value_5:z:0*
T0*
_output_shapes
:�
"simplified_encoder/BN2/SelectV2_15SelectV2"simplified_encoder/BN2/Less_11:z:0!simplified_encoder/BN2/mul_20:z:0!simplified_encoder/BN2/mul_21:z:0*
T0*
_output_shapes
:�
(simplified_encoder/BN2/ReadVariableOp_10ReadVariableOp5simplified_encoder_bn2_relu_3_readvariableop_resource*
_output_shapes
:*
dtype0l
'simplified_encoder/BN2/GreaterEqual_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
%simplified_encoder/BN2/GreaterEqual_3GreaterEqual0simplified_encoder/BN2/ReadVariableOp_10:value:00simplified_encoder/BN2/GreaterEqual_3/y:output:0*
T0*
_output_shapes
:f
$simplified_encoder/BN2/LogicalOr_1/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z�
"simplified_encoder/BN2/LogicalOr_1	LogicalOr)simplified_encoder/BN2/GreaterEqual_3:z:0-simplified_encoder/BN2/LogicalOr_1/y:output:0*
_output_shapes
:c
simplified_encoder/BN2/pow_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
simplified_encoder/BN2/pow_4Pow'simplified_encoder/BN2/pow_4/x:output:0+simplified_encoder/BN2/SelectV2_13:output:0*
T0*
_output_shapes
:c
simplified_encoder/BN2/pow_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
simplified_encoder/BN2/pow_5Pow'simplified_encoder/BN2/pow_5/x:output:0+simplified_encoder/BN2/SelectV2_15:output:0*
T0*
_output_shapes
:k
simplified_encoder/BN2/Neg_12Neg simplified_encoder/BN2/pow_5:z:0*
T0*
_output_shapes
:�
"simplified_encoder/BN2/SelectV2_16SelectV2&simplified_encoder/BN2/LogicalOr_1:z:0 simplified_encoder/BN2/pow_4:z:0!simplified_encoder/BN2/Neg_12:y:0*
T0*
_output_shapes
:v
simplified_encoder/BN2/Neg_13Neg+simplified_encoder/BN2/Relu_3:activations:0*
T0*
_output_shapes
:�
simplified_encoder/BN2/add_20AddV2!simplified_encoder/BN2/Neg_13:y:0+simplified_encoder/BN2/SelectV2_16:output:0*
T0*
_output_shapes
:d
simplified_encoder/BN2/mul_22/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/BN2/mul_22Mul(simplified_encoder/BN2/mul_22/x:output:0!simplified_encoder/BN2/add_20:z:0*
T0*
_output_shapes
:}
%simplified_encoder/BN2/StopGradient_9StopGradient!simplified_encoder/BN2/mul_22:z:0*
T0*
_output_shapes
:�
simplified_encoder/BN2/add_21AddV2+simplified_encoder/BN2/Relu_3:activations:0.simplified_encoder/BN2/StopGradient_9:output:0*
T0*
_output_shapes
:k
&simplified_encoder/BN2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$simplified_encoder/BN2/batchnorm/addAddV2!simplified_encoder/BN2/add_21:z:0/simplified_encoder/BN2/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&simplified_encoder/BN2/batchnorm/RsqrtRsqrt(simplified_encoder/BN2/batchnorm/add:z:0*
T0*
_output_shapes
:�
$simplified_encoder/BN2/batchnorm/mulMul*simplified_encoder/BN2/batchnorm/Rsqrt:y:0 simplified_encoder/BN2/add_5:z:0*
T0*
_output_shapes
:�
&simplified_encoder/BN2/batchnorm/mul_1Mul*simplified_encoder/dense2/BiasAdd:output:0(simplified_encoder/BN2/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
&simplified_encoder/BN2/batchnorm/mul_2Mul!simplified_encoder/BN2/add_15:z:0(simplified_encoder/BN2/batchnorm/mul:z:0*
T0*
_output_shapes
:�
$simplified_encoder/BN2/batchnorm/subSub!simplified_encoder/BN2/add_10:z:0*simplified_encoder/BN2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
&simplified_encoder/BN2/batchnorm/add_1AddV2*simplified_encoder/BN2/batchnorm/mul_1:z:0(simplified_encoder/BN2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������`
simplified_encoder/relu2/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :`
simplified_encoder/relu2/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :�
simplified_encoder/relu2/PowPow'simplified_encoder/relu2/Pow/x:output:0'simplified_encoder/relu2/Pow/y:output:0*
T0*
_output_shapes
: w
simplified_encoder/relu2/CastCast simplified_encoder/relu2/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: b
 simplified_encoder/relu2/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :b
 simplified_encoder/relu2/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : �
simplified_encoder/relu2/Pow_1Pow)simplified_encoder/relu2/Pow_1/x:output:0)simplified_encoder/relu2/Pow_1/y:output:0*
T0*
_output_shapes
: {
simplified_encoder/relu2/Cast_1Cast"simplified_encoder/relu2/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: c
simplified_encoder/relu2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @c
!simplified_encoder/relu2/Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B : �
simplified_encoder/relu2/Cast_2Cast*simplified_encoder/relu2/Cast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: c
simplified_encoder/relu2/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A�
simplified_encoder/relu2/subSub#simplified_encoder/relu2/Cast_2:y:0'simplified_encoder/relu2/sub/y:output:0*
T0*
_output_shapes
: �
simplified_encoder/relu2/Pow_2Pow'simplified_encoder/relu2/Const:output:0 simplified_encoder/relu2/sub:z:0*
T0*
_output_shapes
: �
simplified_encoder/relu2/sub_1Sub#simplified_encoder/relu2/Cast_1:y:0"simplified_encoder/relu2/Pow_2:z:0*
T0*
_output_shapes
: �
"simplified_encoder/relu2/LessEqual	LessEqual*simplified_encoder/BN2/batchnorm/add_1:z:0"simplified_encoder/relu2/sub_1:z:0*
T0*'
_output_shapes
:����������
"simplified_encoder/relu2/LeakyRelu	LeakyRelu*simplified_encoder/BN2/batchnorm/add_1:z:0*'
_output_shapes
:���������*
alpha%  �:�
(simplified_encoder/relu2/ones_like/ShapeShape*simplified_encoder/BN2/batchnorm/add_1:z:0*
T0*
_output_shapes
::��m
(simplified_encoder/relu2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"simplified_encoder/relu2/ones_likeFill1simplified_encoder/relu2/ones_like/Shape:output:01simplified_encoder/relu2/ones_like/Const:output:0*
T0*'
_output_shapes
:����������
simplified_encoder/relu2/sub_2Sub#simplified_encoder/relu2/Cast_1:y:0"simplified_encoder/relu2/Pow_2:z:0*
T0*
_output_shapes
: �
simplified_encoder/relu2/mulMul+simplified_encoder/relu2/ones_like:output:0"simplified_encoder/relu2/sub_2:z:0*
T0*'
_output_shapes
:����������
!simplified_encoder/relu2/SelectV2SelectV2&simplified_encoder/relu2/LessEqual:z:00simplified_encoder/relu2/LeakyRelu:activations:0 simplified_encoder/relu2/mul:z:0*
T0*'
_output_shapes
:����������
simplified_encoder/relu2/mul_1Mul*simplified_encoder/BN2/batchnorm/add_1:z:0!simplified_encoder/relu2/Cast:y:0*
T0*'
_output_shapes
:����������
 simplified_encoder/relu2/truedivRealDiv"simplified_encoder/relu2/mul_1:z:0#simplified_encoder/relu2/Cast_1:y:0*
T0*'
_output_shapes
:���������{
simplified_encoder/relu2/NegNeg$simplified_encoder/relu2/truediv:z:0*
T0*'
_output_shapes
:���������
simplified_encoder/relu2/RoundRound$simplified_encoder/relu2/truediv:z:0*
T0*'
_output_shapes
:����������
simplified_encoder/relu2/addAddV2 simplified_encoder/relu2/Neg:y:0"simplified_encoder/relu2/Round:y:0*
T0*'
_output_shapes
:����������
%simplified_encoder/relu2/StopGradientStopGradient simplified_encoder/relu2/add:z:0*
T0*'
_output_shapes
:����������
simplified_encoder/relu2/add_1AddV2$simplified_encoder/relu2/truediv:z:0.simplified_encoder/relu2/StopGradient:output:0*
T0*'
_output_shapes
:����������
"simplified_encoder/relu2/truediv_1RealDiv"simplified_encoder/relu2/add_1:z:0!simplified_encoder/relu2/Cast:y:0*
T0*'
_output_shapes
:���������i
$simplified_encoder/relu2/truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"simplified_encoder/relu2/truediv_2RealDiv-simplified_encoder/relu2/truediv_2/x:output:0!simplified_encoder/relu2/Cast:y:0*
T0*
_output_shapes
: e
 simplified_encoder/relu2/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/relu2/sub_3Sub)simplified_encoder/relu2/sub_3/x:output:0&simplified_encoder/relu2/truediv_2:z:0*
T0*
_output_shapes
: �
.simplified_encoder/relu2/clip_by_value/MinimumMinimum&simplified_encoder/relu2/truediv_1:z:0"simplified_encoder/relu2/sub_3:z:0*
T0*'
_output_shapes
:���������m
(simplified_encoder/relu2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
&simplified_encoder/relu2/clip_by_valueMaximum2simplified_encoder/relu2/clip_by_value/Minimum:z:01simplified_encoder/relu2/clip_by_value/y:output:0*
T0*'
_output_shapes
:����������
simplified_encoder/relu2/mul_2Mul#simplified_encoder/relu2/Cast_1:y:0*simplified_encoder/relu2/clip_by_value:z:0*
T0*'
_output_shapes
:���������e
 simplified_encoder/relu2/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �:�
simplified_encoder/relu2/mul_3Mul)simplified_encoder/relu2/mul_3/x:output:0!simplified_encoder/relu2/Cast:y:0*
T0*
_output_shapes
: i
$simplified_encoder/relu2/truediv_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"simplified_encoder/relu2/truediv_3RealDiv-simplified_encoder/relu2/truediv_3/x:output:0"simplified_encoder/relu2/mul_3:z:0*
T0*
_output_shapes
: e
 simplified_encoder/relu2/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �:�
simplified_encoder/relu2/mul_4Mul#simplified_encoder/relu2/Cast_1:y:0)simplified_encoder/relu2/mul_4/y:output:0*
T0*
_output_shapes
: e
 simplified_encoder/relu2/mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �:�
simplified_encoder/relu2/mul_5Mul$simplified_encoder/relu2/truediv:z:0)simplified_encoder/relu2/mul_5/y:output:0*
T0*'
_output_shapes
:���������{
simplified_encoder/relu2/Neg_1Neg"simplified_encoder/relu2/mul_5:z:0*
T0*'
_output_shapes
:���������
 simplified_encoder/relu2/Round_1Round"simplified_encoder/relu2/mul_5:z:0*
T0*'
_output_shapes
:����������
simplified_encoder/relu2/add_2AddV2"simplified_encoder/relu2/Neg_1:y:0$simplified_encoder/relu2/Round_1:y:0*
T0*'
_output_shapes
:����������
'simplified_encoder/relu2/StopGradient_1StopGradient"simplified_encoder/relu2/add_2:z:0*
T0*'
_output_shapes
:����������
simplified_encoder/relu2/add_3AddV2"simplified_encoder/relu2/mul_5:z:00simplified_encoder/relu2/StopGradient_1:output:0*
T0*'
_output_shapes
:����������
simplified_encoder/relu2/mul_6Mul"simplified_encoder/relu2/add_3:z:0&simplified_encoder/relu2/truediv_3:z:0*
T0*'
_output_shapes
:���������w
2simplified_encoder/relu2/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
0simplified_encoder/relu2/clip_by_value_1/MinimumMinimum"simplified_encoder/relu2/mul_6:z:0;simplified_encoder/relu2/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:���������o
*simplified_encoder/relu2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
(simplified_encoder/relu2/clip_by_value_1Maximum4simplified_encoder/relu2/clip_by_value_1/Minimum:z:03simplified_encoder/relu2/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:����������
simplified_encoder/relu2/mul_7Mul"simplified_encoder/relu2/mul_4:z:0,simplified_encoder/relu2/clip_by_value_1:z:0*
T0*'
_output_shapes
:����������
simplified_encoder/relu2/add_4AddV2"simplified_encoder/relu2/mul_2:z:0"simplified_encoder/relu2/mul_7:z:0*
T0*'
_output_shapes
:����������
simplified_encoder/relu2/Neg_2Neg*simplified_encoder/relu2/SelectV2:output:0*
T0*'
_output_shapes
:����������
simplified_encoder/relu2/add_5AddV2"simplified_encoder/relu2/Neg_2:y:0"simplified_encoder/relu2/add_4:z:0*
T0*'
_output_shapes
:���������e
 simplified_encoder/relu2/mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/relu2/mul_8Mul)simplified_encoder/relu2/mul_8/x:output:0"simplified_encoder/relu2/add_5:z:0*
T0*'
_output_shapes
:����������
'simplified_encoder/relu2/StopGradient_2StopGradient"simplified_encoder/relu2/mul_8:z:0*
T0*'
_output_shapes
:����������
simplified_encoder/relu2/add_6AddV2*simplified_encoder/relu2/SelectV2:output:00simplified_encoder/relu2/StopGradient_2:output:0*
T0*'
_output_shapes
:���������a
simplified_encoder/z_mean/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :a
simplified_encoder/z_mean/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :�
simplified_encoder/z_mean/PowPow(simplified_encoder/z_mean/Pow/x:output:0(simplified_encoder/z_mean/Pow/y:output:0*
T0*
_output_shapes
: y
simplified_encoder/z_mean/CastCast!simplified_encoder/z_mean/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: �
(simplified_encoder/z_mean/ReadVariableOpReadVariableOp1simplified_encoder_z_mean_readvariableop_resource*
_output_shapes

:*
dtype0d
simplified_encoder/z_mean/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   D�
simplified_encoder/z_mean/mulMul0simplified_encoder/z_mean/ReadVariableOp:value:0(simplified_encoder/z_mean/mul/y:output:0*
T0*
_output_shapes

:�
!simplified_encoder/z_mean/truedivRealDiv!simplified_encoder/z_mean/mul:z:0"simplified_encoder/z_mean/Cast:y:0*
T0*
_output_shapes

:t
simplified_encoder/z_mean/NegNeg%simplified_encoder/z_mean/truediv:z:0*
T0*
_output_shapes

:x
simplified_encoder/z_mean/RoundRound%simplified_encoder/z_mean/truediv:z:0*
T0*
_output_shapes

:�
simplified_encoder/z_mean/addAddV2!simplified_encoder/z_mean/Neg:y:0#simplified_encoder/z_mean/Round:y:0*
T0*
_output_shapes

:�
&simplified_encoder/z_mean/StopGradientStopGradient!simplified_encoder/z_mean/add:z:0*
T0*
_output_shapes

:�
simplified_encoder/z_mean/add_1AddV2%simplified_encoder/z_mean/truediv:z:0/simplified_encoder/z_mean/StopGradient:output:0*
T0*
_output_shapes

:v
1simplified_encoder/z_mean/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ��C�
/simplified_encoder/z_mean/clip_by_value/MinimumMinimum#simplified_encoder/z_mean/add_1:z:0:simplified_encoder/z_mean/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:n
)simplified_encoder/z_mean/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
'simplified_encoder/z_mean/clip_by_valueMaximum3simplified_encoder/z_mean/clip_by_value/Minimum:z:02simplified_encoder/z_mean/clip_by_value/y:output:0*
T0*
_output_shapes

:�
simplified_encoder/z_mean/mul_1Mul"simplified_encoder/z_mean/Cast:y:0+simplified_encoder/z_mean/clip_by_value:z:0*
T0*
_output_shapes

:j
%simplified_encoder/z_mean/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   D�
#simplified_encoder/z_mean/truediv_1RealDiv#simplified_encoder/z_mean/mul_1:z:0.simplified_encoder/z_mean/truediv_1/y:output:0*
T0*
_output_shapes

:f
!simplified_encoder/z_mean/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/z_mean/mul_2Mul*simplified_encoder/z_mean/mul_2/x:output:0'simplified_encoder/z_mean/truediv_1:z:0*
T0*
_output_shapes

:�
*simplified_encoder/z_mean/ReadVariableOp_1ReadVariableOp1simplified_encoder_z_mean_readvariableop_resource*
_output_shapes

:*
dtype0�
simplified_encoder/z_mean/Neg_1Neg2simplified_encoder/z_mean/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
simplified_encoder/z_mean/add_2AddV2#simplified_encoder/z_mean/Neg_1:y:0#simplified_encoder/z_mean/mul_2:z:0*
T0*
_output_shapes

:f
!simplified_encoder/z_mean/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/z_mean/mul_3Mul*simplified_encoder/z_mean/mul_3/x:output:0#simplified_encoder/z_mean/add_2:z:0*
T0*
_output_shapes

:�
(simplified_encoder/z_mean/StopGradient_1StopGradient#simplified_encoder/z_mean/mul_3:z:0*
T0*
_output_shapes

:�
*simplified_encoder/z_mean/ReadVariableOp_2ReadVariableOp1simplified_encoder_z_mean_readvariableop_resource*
_output_shapes

:*
dtype0�
simplified_encoder/z_mean/add_3AddV22simplified_encoder/z_mean/ReadVariableOp_2:value:01simplified_encoder/z_mean/StopGradient_1:output:0*
T0*
_output_shapes

:�
 simplified_encoder/z_mean/MatMulMatMul"simplified_encoder/relu2/add_6:z:0#simplified_encoder/z_mean/add_3:z:0*
T0*'
_output_shapes
:���������c
!simplified_encoder/z_mean/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :c
!simplified_encoder/z_mean/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
simplified_encoder/z_mean/Pow_1Pow*simplified_encoder/z_mean/Pow_1/x:output:0*simplified_encoder/z_mean/Pow_1/y:output:0*
T0*
_output_shapes
: }
 simplified_encoder/z_mean/Cast_1Cast#simplified_encoder/z_mean/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
*simplified_encoder/z_mean/ReadVariableOp_3ReadVariableOp3simplified_encoder_z_mean_readvariableop_3_resource*
_output_shapes
:*
dtype0f
!simplified_encoder/z_mean/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   D�
simplified_encoder/z_mean/mul_4Mul2simplified_encoder/z_mean/ReadVariableOp_3:value:0*simplified_encoder/z_mean/mul_4/y:output:0*
T0*
_output_shapes
:�
#simplified_encoder/z_mean/truediv_2RealDiv#simplified_encoder/z_mean/mul_4:z:0$simplified_encoder/z_mean/Cast_1:y:0*
T0*
_output_shapes
:t
simplified_encoder/z_mean/Neg_2Neg'simplified_encoder/z_mean/truediv_2:z:0*
T0*
_output_shapes
:x
!simplified_encoder/z_mean/Round_1Round'simplified_encoder/z_mean/truediv_2:z:0*
T0*
_output_shapes
:�
simplified_encoder/z_mean/add_4AddV2#simplified_encoder/z_mean/Neg_2:y:0%simplified_encoder/z_mean/Round_1:y:0*
T0*
_output_shapes
:�
(simplified_encoder/z_mean/StopGradient_2StopGradient#simplified_encoder/z_mean/add_4:z:0*
T0*
_output_shapes
:�
simplified_encoder/z_mean/add_5AddV2'simplified_encoder/z_mean/truediv_2:z:01simplified_encoder/z_mean/StopGradient_2:output:0*
T0*
_output_shapes
:x
3simplified_encoder/z_mean/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ��C�
1simplified_encoder/z_mean/clip_by_value_1/MinimumMinimum#simplified_encoder/z_mean/add_5:z:0<simplified_encoder/z_mean/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:p
+simplified_encoder/z_mean/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
)simplified_encoder/z_mean/clip_by_value_1Maximum5simplified_encoder/z_mean/clip_by_value_1/Minimum:z:04simplified_encoder/z_mean/clip_by_value_1/y:output:0*
T0*
_output_shapes
:�
simplified_encoder/z_mean/mul_5Mul$simplified_encoder/z_mean/Cast_1:y:0-simplified_encoder/z_mean/clip_by_value_1:z:0*
T0*
_output_shapes
:j
%simplified_encoder/z_mean/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   D�
#simplified_encoder/z_mean/truediv_3RealDiv#simplified_encoder/z_mean/mul_5:z:0.simplified_encoder/z_mean/truediv_3/y:output:0*
T0*
_output_shapes
:f
!simplified_encoder/z_mean/mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/z_mean/mul_6Mul*simplified_encoder/z_mean/mul_6/x:output:0'simplified_encoder/z_mean/truediv_3:z:0*
T0*
_output_shapes
:�
*simplified_encoder/z_mean/ReadVariableOp_4ReadVariableOp3simplified_encoder_z_mean_readvariableop_3_resource*
_output_shapes
:*
dtype0
simplified_encoder/z_mean/Neg_3Neg2simplified_encoder/z_mean/ReadVariableOp_4:value:0*
T0*
_output_shapes
:�
simplified_encoder/z_mean/add_6AddV2#simplified_encoder/z_mean/Neg_3:y:0#simplified_encoder/z_mean/mul_6:z:0*
T0*
_output_shapes
:f
!simplified_encoder/z_mean/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
simplified_encoder/z_mean/mul_7Mul*simplified_encoder/z_mean/mul_7/x:output:0#simplified_encoder/z_mean/add_6:z:0*
T0*
_output_shapes
:�
(simplified_encoder/z_mean/StopGradient_3StopGradient#simplified_encoder/z_mean/mul_7:z:0*
T0*
_output_shapes
:�
*simplified_encoder/z_mean/ReadVariableOp_5ReadVariableOp3simplified_encoder_z_mean_readvariableop_3_resource*
_output_shapes
:*
dtype0�
simplified_encoder/z_mean/add_7AddV22simplified_encoder/z_mean/ReadVariableOp_5:value:01simplified_encoder/z_mean/StopGradient_3:output:0*
T0*
_output_shapes
:�
!simplified_encoder/z_mean/BiasAddBiasAdd*simplified_encoder/z_mean/MatMul:product:0#simplified_encoder/z_mean/add_7:z:0*
T0*'
_output_shapes
:���������y
IdentityIdentity*simplified_encoder/z_mean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^simplified_encoder/BN1/Abs_1/ReadVariableOp,^simplified_encoder/BN1/Abs_3/ReadVariableOp&^simplified_encoder/BN1/ReadVariableOp(^simplified_encoder/BN1/ReadVariableOp_1)^simplified_encoder/BN1/ReadVariableOp_10(^simplified_encoder/BN1/ReadVariableOp_2(^simplified_encoder/BN1/ReadVariableOp_3(^simplified_encoder/BN1/ReadVariableOp_4(^simplified_encoder/BN1/ReadVariableOp_5(^simplified_encoder/BN1/ReadVariableOp_6(^simplified_encoder/BN1/ReadVariableOp_7(^simplified_encoder/BN1/ReadVariableOp_8(^simplified_encoder/BN1/ReadVariableOp_9+^simplified_encoder/BN1/Relu/ReadVariableOp-^simplified_encoder/BN1/Relu_1/ReadVariableOp-^simplified_encoder/BN1/Relu_3/ReadVariableOp-^simplified_encoder/BN1/Relu_4/ReadVariableOp,^simplified_encoder/BN2/Abs_1/ReadVariableOp,^simplified_encoder/BN2/Abs_3/ReadVariableOp&^simplified_encoder/BN2/ReadVariableOp(^simplified_encoder/BN2/ReadVariableOp_1)^simplified_encoder/BN2/ReadVariableOp_10(^simplified_encoder/BN2/ReadVariableOp_2(^simplified_encoder/BN2/ReadVariableOp_3(^simplified_encoder/BN2/ReadVariableOp_4(^simplified_encoder/BN2/ReadVariableOp_5(^simplified_encoder/BN2/ReadVariableOp_6(^simplified_encoder/BN2/ReadVariableOp_7(^simplified_encoder/BN2/ReadVariableOp_8(^simplified_encoder/BN2/ReadVariableOp_9+^simplified_encoder/BN2/Relu/ReadVariableOp-^simplified_encoder/BN2/Relu_1/ReadVariableOp-^simplified_encoder/BN2/Relu_3/ReadVariableOp-^simplified_encoder/BN2/Relu_4/ReadVariableOp)^simplified_encoder/dense1/ReadVariableOp+^simplified_encoder/dense1/ReadVariableOp_1+^simplified_encoder/dense1/ReadVariableOp_2+^simplified_encoder/dense1/ReadVariableOp_3+^simplified_encoder/dense1/ReadVariableOp_4+^simplified_encoder/dense1/ReadVariableOp_5)^simplified_encoder/dense2/ReadVariableOp+^simplified_encoder/dense2/ReadVariableOp_1+^simplified_encoder/dense2/ReadVariableOp_2+^simplified_encoder/dense2/ReadVariableOp_3+^simplified_encoder/dense2/ReadVariableOp_4+^simplified_encoder/dense2/ReadVariableOp_5)^simplified_encoder/z_mean/ReadVariableOp+^simplified_encoder/z_mean/ReadVariableOp_1+^simplified_encoder/z_mean/ReadVariableOp_2+^simplified_encoder/z_mean/ReadVariableOp_3+^simplified_encoder/z_mean/ReadVariableOp_4+^simplified_encoder/z_mean/ReadVariableOp_5*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������,: : : : : : : : : : : : : : 2Z
+simplified_encoder/BN1/Abs_1/ReadVariableOp+simplified_encoder/BN1/Abs_1/ReadVariableOp2Z
+simplified_encoder/BN1/Abs_3/ReadVariableOp+simplified_encoder/BN1/Abs_3/ReadVariableOp2N
%simplified_encoder/BN1/ReadVariableOp%simplified_encoder/BN1/ReadVariableOp2R
'simplified_encoder/BN1/ReadVariableOp_1'simplified_encoder/BN1/ReadVariableOp_12T
(simplified_encoder/BN1/ReadVariableOp_10(simplified_encoder/BN1/ReadVariableOp_102R
'simplified_encoder/BN1/ReadVariableOp_2'simplified_encoder/BN1/ReadVariableOp_22R
'simplified_encoder/BN1/ReadVariableOp_3'simplified_encoder/BN1/ReadVariableOp_32R
'simplified_encoder/BN1/ReadVariableOp_4'simplified_encoder/BN1/ReadVariableOp_42R
'simplified_encoder/BN1/ReadVariableOp_5'simplified_encoder/BN1/ReadVariableOp_52R
'simplified_encoder/BN1/ReadVariableOp_6'simplified_encoder/BN1/ReadVariableOp_62R
'simplified_encoder/BN1/ReadVariableOp_7'simplified_encoder/BN1/ReadVariableOp_72R
'simplified_encoder/BN1/ReadVariableOp_8'simplified_encoder/BN1/ReadVariableOp_82R
'simplified_encoder/BN1/ReadVariableOp_9'simplified_encoder/BN1/ReadVariableOp_92X
*simplified_encoder/BN1/Relu/ReadVariableOp*simplified_encoder/BN1/Relu/ReadVariableOp2\
,simplified_encoder/BN1/Relu_1/ReadVariableOp,simplified_encoder/BN1/Relu_1/ReadVariableOp2\
,simplified_encoder/BN1/Relu_3/ReadVariableOp,simplified_encoder/BN1/Relu_3/ReadVariableOp2\
,simplified_encoder/BN1/Relu_4/ReadVariableOp,simplified_encoder/BN1/Relu_4/ReadVariableOp2Z
+simplified_encoder/BN2/Abs_1/ReadVariableOp+simplified_encoder/BN2/Abs_1/ReadVariableOp2Z
+simplified_encoder/BN2/Abs_3/ReadVariableOp+simplified_encoder/BN2/Abs_3/ReadVariableOp2N
%simplified_encoder/BN2/ReadVariableOp%simplified_encoder/BN2/ReadVariableOp2R
'simplified_encoder/BN2/ReadVariableOp_1'simplified_encoder/BN2/ReadVariableOp_12T
(simplified_encoder/BN2/ReadVariableOp_10(simplified_encoder/BN2/ReadVariableOp_102R
'simplified_encoder/BN2/ReadVariableOp_2'simplified_encoder/BN2/ReadVariableOp_22R
'simplified_encoder/BN2/ReadVariableOp_3'simplified_encoder/BN2/ReadVariableOp_32R
'simplified_encoder/BN2/ReadVariableOp_4'simplified_encoder/BN2/ReadVariableOp_42R
'simplified_encoder/BN2/ReadVariableOp_5'simplified_encoder/BN2/ReadVariableOp_52R
'simplified_encoder/BN2/ReadVariableOp_6'simplified_encoder/BN2/ReadVariableOp_62R
'simplified_encoder/BN2/ReadVariableOp_7'simplified_encoder/BN2/ReadVariableOp_72R
'simplified_encoder/BN2/ReadVariableOp_8'simplified_encoder/BN2/ReadVariableOp_82R
'simplified_encoder/BN2/ReadVariableOp_9'simplified_encoder/BN2/ReadVariableOp_92X
*simplified_encoder/BN2/Relu/ReadVariableOp*simplified_encoder/BN2/Relu/ReadVariableOp2\
,simplified_encoder/BN2/Relu_1/ReadVariableOp,simplified_encoder/BN2/Relu_1/ReadVariableOp2\
,simplified_encoder/BN2/Relu_3/ReadVariableOp,simplified_encoder/BN2/Relu_3/ReadVariableOp2\
,simplified_encoder/BN2/Relu_4/ReadVariableOp,simplified_encoder/BN2/Relu_4/ReadVariableOp2T
(simplified_encoder/dense1/ReadVariableOp(simplified_encoder/dense1/ReadVariableOp2X
*simplified_encoder/dense1/ReadVariableOp_1*simplified_encoder/dense1/ReadVariableOp_12X
*simplified_encoder/dense1/ReadVariableOp_2*simplified_encoder/dense1/ReadVariableOp_22X
*simplified_encoder/dense1/ReadVariableOp_3*simplified_encoder/dense1/ReadVariableOp_32X
*simplified_encoder/dense1/ReadVariableOp_4*simplified_encoder/dense1/ReadVariableOp_42X
*simplified_encoder/dense1/ReadVariableOp_5*simplified_encoder/dense1/ReadVariableOp_52T
(simplified_encoder/dense2/ReadVariableOp(simplified_encoder/dense2/ReadVariableOp2X
*simplified_encoder/dense2/ReadVariableOp_1*simplified_encoder/dense2/ReadVariableOp_12X
*simplified_encoder/dense2/ReadVariableOp_2*simplified_encoder/dense2/ReadVariableOp_22X
*simplified_encoder/dense2/ReadVariableOp_3*simplified_encoder/dense2/ReadVariableOp_32X
*simplified_encoder/dense2/ReadVariableOp_4*simplified_encoder/dense2/ReadVariableOp_42X
*simplified_encoder/dense2/ReadVariableOp_5*simplified_encoder/dense2/ReadVariableOp_52T
(simplified_encoder/z_mean/ReadVariableOp(simplified_encoder/z_mean/ReadVariableOp2X
*simplified_encoder/z_mean/ReadVariableOp_1*simplified_encoder/z_mean/ReadVariableOp_12X
*simplified_encoder/z_mean/ReadVariableOp_2*simplified_encoder/z_mean/ReadVariableOp_22X
*simplified_encoder/z_mean/ReadVariableOp_3*simplified_encoder/z_mean/ReadVariableOp_32X
*simplified_encoder/z_mean/ReadVariableOp_4*simplified_encoder/z_mean/ReadVariableOp_42X
*simplified_encoder/z_mean/ReadVariableOp_5*simplified_encoder/z_mean/ReadVariableOp_5:O K
'
_output_shapes
:���������,
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
%__inference_signature_wrapper_8205179

inputs
unknown:, 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_8203144o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������,: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������,
 
_user_specified_nameinputs:'#
!
_user_specified_name	8205149:'#
!
_user_specified_name	8205151:'#
!
_user_specified_name	8205153:'#
!
_user_specified_name	8205155:'#
!
_user_specified_name	8205157:'#
!
_user_specified_name	8205159:'#
!
_user_specified_name	8205161:'#
!
_user_specified_name	8205163:'	#
!
_user_specified_name	8205165:'
#
!
_user_specified_name	8205167:'#
!
_user_specified_name	8205169:'#
!
_user_specified_name	8205171:'#
!
_user_specified_name	8205173:'#
!
_user_specified_name	8205175
�C
�
#__inference__traced_restore_8207193
file_prefix0
assignvariableop_dense1_kernel:, ,
assignvariableop_1_dense1_bias: *
assignvariableop_2_bn1_gamma: )
assignvariableop_3_bn1_beta: 0
"assignvariableop_4_bn1_moving_mean: 4
&assignvariableop_5_bn1_moving_variance: 2
 assignvariableop_6_dense2_kernel: ,
assignvariableop_7_dense2_bias:*
assignvariableop_8_bn2_gamma:)
assignvariableop_9_bn2_beta:1
#assignvariableop_10_bn2_moving_mean:5
'assignvariableop_11_bn2_moving_variance:3
!assignvariableop_12_z_mean_kernel:-
assignvariableop_13_z_mean_bias:
identity_15��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*P
_output_shapes>
<:::::::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense1_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense1_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_bn1_gammaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_bn1_betaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_bn1_moving_meanIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp&assignvariableop_5_bn1_moving_varianceIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp assignvariableop_6_dense2_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense2_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_bn2_gammaIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_bn2_betaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp#assignvariableop_10_bn2_moving_meanIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp'assignvariableop_11_bn2_moving_varianceIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp!assignvariableop_12_z_mean_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_z_mean_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_15IdentityIdentity_14:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_15Identity_15:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
: : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_user_specified_namedense1/kernel:+'
%
_user_specified_namedense1/bias:)%
#
_user_specified_name	BN1/gamma:($
"
_user_specified_name
BN1/beta:/+
)
_user_specified_nameBN1/moving_mean:3/
-
_user_specified_nameBN1/moving_variance:-)
'
_user_specified_namedense2/kernel:+'
%
_user_specified_namedense2/bias:)	%
#
_user_specified_name	BN2/gamma:(
$
"
_user_specified_name
BN2/beta:/+
)
_user_specified_nameBN2/moving_mean:3/
-
_user_specified_nameBN2/moving_variance:-)
'
_user_specified_namez_mean/kernel:+'
%
_user_specified_namez_mean/bias
�/
^
B__inference_relu2_layer_call_and_return_conditional_losses_8204295

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B : Q
Cast_2CastCast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `AG
subSub
Cast_2:y:0sub/y:output:0*
T0*
_output_shapes
: F
Pow_2PowConst:output:0sub:z:0*
T0*
_output_shapes
: D
sub_1Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
	LessEqual	LessEqualinputs	sub_1:z:0*
T0*'
_output_shapes
:���������W
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:���������*
alpha%  �:S
ones_like/ShapeShapeinputs*
T0*
_output_shapes
::��T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:���������D
sub_2Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
mulMulones_like:output:0	sub_2:z:0*
T0*'
_output_shapes
:���������w
SelectV2SelectV2LessEqual:z:0LeakyRelu:activations:0mul:z:0*
T0*'
_output_shapes
:���������P
mul_1MulinputsCast:y:0*
T0*'
_output_shapes
:���������[
truedivRealDiv	mul_1:z:0
Cast_1:y:0*
T0*'
_output_shapes
:���������I
NegNegtruediv:z:0*
T0*'
_output_shapes
:���������M
RoundRoundtruediv:z:0*
T0*'
_output_shapes
:���������R
addAddV2Neg:y:0	Round:y:0*
T0*'
_output_shapes
:���������W
StopGradientStopGradientadd:z:0*
T0*'
_output_shapes
:���������d
add_1AddV2truediv:z:0StopGradient:output:0*
T0*'
_output_shapes
:���������[
	truediv_1RealDiv	add_1:z:0Cast:y:0*
T0*'
_output_shapes
:���������P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
	truediv_2RealDivtruediv_2/x:output:0Cast:y:0*
T0*
_output_shapes
: L
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_3Subsub_3/x:output:0truediv_2:z:0*
T0*
_output_shapes
: l
clip_by_value/MinimumMinimumtruediv_1:z:0	sub_3:z:0*
T0*'
_output_shapes
:���������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������]
mul_2Mul
Cast_1:y:0clip_by_value:z:0*
T0*'
_output_shapes
:���������L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �:I
mul_3Mulmul_3/x:output:0Cast:y:0*
T0*
_output_shapes
: P
truediv_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
	truediv_3RealDivtruediv_3/x:output:0	mul_3:z:0*
T0*
_output_shapes
: L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �:K
mul_4Mul
Cast_1:y:0mul_4/y:output:0*
T0*
_output_shapes
: L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �:]
mul_5Multruediv:z:0mul_5/y:output:0*
T0*'
_output_shapes
:���������I
Neg_1Neg	mul_5:z:0*
T0*'
_output_shapes
:���������M
Round_1Round	mul_5:z:0*
T0*'
_output_shapes
:���������X
add_2AddV2	Neg_1:y:0Round_1:y:0*
T0*'
_output_shapes
:���������[
StopGradient_1StopGradient	add_2:z:0*
T0*'
_output_shapes
:���������d
add_3AddV2	mul_5:z:0StopGradient_1:output:0*
T0*'
_output_shapes
:���������X
mul_6Mul	add_3:z:0truediv_3:z:0*
T0*'
_output_shapes
:���������^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_value_1/MinimumMinimum	mul_6:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:���������V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:���������^
mul_7Mul	mul_4:z:0clip_by_value_1:z:0*
T0*'
_output_shapes
:���������V
add_4AddV2	mul_2:z:0	mul_7:z:0*
T0*'
_output_shapes
:���������Q
Neg_2NegSelectV2:output:0*
T0*'
_output_shapes
:���������V
add_5AddV2	Neg_2:y:0	add_4:z:0*
T0*'
_output_shapes
:���������L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
mul_8Mulmul_8/x:output:0	add_5:z:0*
T0*'
_output_shapes
:���������[
StopGradient_2StopGradient	mul_8:z:0*
T0*'
_output_shapes
:���������l
add_6AddV2SelectV2:output:0StopGradient_2:output:0*
T0*'
_output_shapes
:���������Q
IdentityIdentity	add_6:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�/
^
B__inference_relu1_layer_call_and_return_conditional_losses_8203720

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B : Q
Cast_2CastCast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `AG
subSub
Cast_2:y:0sub/y:output:0*
T0*
_output_shapes
: F
Pow_2PowConst:output:0sub:z:0*
T0*
_output_shapes
: D
sub_1Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
	LessEqual	LessEqualinputs	sub_1:z:0*
T0*'
_output_shapes
:��������� W
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:��������� *
alpha%  �:S
ones_like/ShapeShapeinputs*
T0*
_output_shapes
::��T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:��������� D
sub_2Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
mulMulones_like:output:0	sub_2:z:0*
T0*'
_output_shapes
:��������� w
SelectV2SelectV2LessEqual:z:0LeakyRelu:activations:0mul:z:0*
T0*'
_output_shapes
:��������� P
mul_1MulinputsCast:y:0*
T0*'
_output_shapes
:��������� [
truedivRealDiv	mul_1:z:0
Cast_1:y:0*
T0*'
_output_shapes
:��������� I
NegNegtruediv:z:0*
T0*'
_output_shapes
:��������� M
RoundRoundtruediv:z:0*
T0*'
_output_shapes
:��������� R
addAddV2Neg:y:0	Round:y:0*
T0*'
_output_shapes
:��������� W
StopGradientStopGradientadd:z:0*
T0*'
_output_shapes
:��������� d
add_1AddV2truediv:z:0StopGradient:output:0*
T0*'
_output_shapes
:��������� [
	truediv_1RealDiv	add_1:z:0Cast:y:0*
T0*'
_output_shapes
:��������� P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
	truediv_2RealDivtruediv_2/x:output:0Cast:y:0*
T0*
_output_shapes
: L
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_3Subsub_3/x:output:0truediv_2:z:0*
T0*
_output_shapes
: l
clip_by_value/MinimumMinimumtruediv_1:z:0	sub_3:z:0*
T0*'
_output_shapes
:��������� T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:��������� ]
mul_2Mul
Cast_1:y:0clip_by_value:z:0*
T0*'
_output_shapes
:��������� L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �:I
mul_3Mulmul_3/x:output:0Cast:y:0*
T0*
_output_shapes
: P
truediv_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
	truediv_3RealDivtruediv_3/x:output:0	mul_3:z:0*
T0*
_output_shapes
: L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �:K
mul_4Mul
Cast_1:y:0mul_4/y:output:0*
T0*
_output_shapes
: L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �:]
mul_5Multruediv:z:0mul_5/y:output:0*
T0*'
_output_shapes
:��������� I
Neg_1Neg	mul_5:z:0*
T0*'
_output_shapes
:��������� M
Round_1Round	mul_5:z:0*
T0*'
_output_shapes
:��������� X
add_2AddV2	Neg_1:y:0Round_1:y:0*
T0*'
_output_shapes
:��������� [
StopGradient_1StopGradient	add_2:z:0*
T0*'
_output_shapes
:��������� d
add_3AddV2	mul_5:z:0StopGradient_1:output:0*
T0*'
_output_shapes
:��������� X
mul_6Mul	add_3:z:0truediv_3:z:0*
T0*'
_output_shapes
:��������� ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_value_1/MinimumMinimum	mul_6:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:��������� V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:��������� ^
mul_7Mul	mul_4:z:0clip_by_value_1:z:0*
T0*'
_output_shapes
:��������� V
add_4AddV2	mul_2:z:0	mul_7:z:0*
T0*'
_output_shapes
:��������� Q
Neg_2NegSelectV2:output:0*
T0*'
_output_shapes
:��������� V
add_5AddV2	Neg_2:y:0	add_4:z:0*
T0*'
_output_shapes
:��������� L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
mul_8Mulmul_8/x:output:0	add_5:z:0*
T0*'
_output_shapes
:��������� [
StopGradient_2StopGradient	mul_8:z:0*
T0*'
_output_shapes
:��������� l
add_6AddV2SelectV2:output:0StopGradient_2:output:0*
T0*'
_output_shapes
:��������� Q
IdentityIdentity	add_6:z:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�2
�
C__inference_dense2_layer_call_and_return_conditional_losses_8206146

inputs)
readvariableop_resource: '
readvariableop_3_resource:
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

: *
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   D[
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes

: N
truedivRealDivmul:z:0Cast:y:0*
T0*
_output_shapes

: @
NegNegtruediv:z:0*
T0*
_output_shapes

: D
RoundRoundtruediv:z:0*
T0*
_output_shapes

: I
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

: N
StopGradientStopGradientadd:z:0*
T0*
_output_shapes

: [
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

: \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ��Cv
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

: T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

: R
mul_1MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

: P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   D^
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*
_output_shapes

: L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*
_output_shapes

: h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

: *
dtype0O
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes

: M
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*
_output_shapes

: L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*
_output_shapes

: R
StopGradient_1StopGradient	mul_3:z:0*
T0*
_output_shapes

: h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

: *
dtype0j
add_3AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*
_output_shapes

: U
MatMulMatMulinputs	add_3:z:0*
T0*'
_output_shapes
:���������I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   D]
mul_4MulReadVariableOp_3:value:0mul_4/y:output:0*
T0*
_output_shapes
:P
	truediv_2RealDiv	mul_4:z:0
Cast_1:y:0*
T0*
_output_shapes
:@
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_4AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_4:z:0*
T0*
_output_shapes
:[
add_5AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ��Cv
clip_by_value_1/MinimumMinimum	add_5:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:R
mul_5Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   DZ
	truediv_3RealDiv	mul_5:z:0truediv_3/y:output:0*
T0*
_output_shapes
:L
mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_6Mulmul_6/x:output:0truediv_3:z:0*
T0*
_output_shapes
:f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0K
Neg_3NegReadVariableOp_4:value:0*
T0*
_output_shapes
:I
add_6AddV2	Neg_3:y:0	mul_6:z:0*
T0*
_output_shapes
:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_7Mulmul_7/x:output:0	add_6:z:0*
T0*
_output_shapes
:N
StopGradient_3StopGradient	mul_7:z:0*
T0*
_output_shapes
:f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0f
add_7AddV2ReadVariableOp_5:value:0StopGradient_3:output:0*
T0*
_output_shapes
:a
BiasAddBiasAddMatMul:product:0	add_7:z:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_5:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�/
^
B__inference_relu1_layer_call_and_return_conditional_losses_8206069

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B : Q
Cast_2CastCast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `AG
subSub
Cast_2:y:0sub/y:output:0*
T0*
_output_shapes
: F
Pow_2PowConst:output:0sub:z:0*
T0*
_output_shapes
: D
sub_1Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
	LessEqual	LessEqualinputs	sub_1:z:0*
T0*'
_output_shapes
:��������� W
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:��������� *
alpha%  �:S
ones_like/ShapeShapeinputs*
T0*
_output_shapes
::��T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:��������� D
sub_2Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
mulMulones_like:output:0	sub_2:z:0*
T0*'
_output_shapes
:��������� w
SelectV2SelectV2LessEqual:z:0LeakyRelu:activations:0mul:z:0*
T0*'
_output_shapes
:��������� P
mul_1MulinputsCast:y:0*
T0*'
_output_shapes
:��������� [
truedivRealDiv	mul_1:z:0
Cast_1:y:0*
T0*'
_output_shapes
:��������� I
NegNegtruediv:z:0*
T0*'
_output_shapes
:��������� M
RoundRoundtruediv:z:0*
T0*'
_output_shapes
:��������� R
addAddV2Neg:y:0	Round:y:0*
T0*'
_output_shapes
:��������� W
StopGradientStopGradientadd:z:0*
T0*'
_output_shapes
:��������� d
add_1AddV2truediv:z:0StopGradient:output:0*
T0*'
_output_shapes
:��������� [
	truediv_1RealDiv	add_1:z:0Cast:y:0*
T0*'
_output_shapes
:��������� P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
	truediv_2RealDivtruediv_2/x:output:0Cast:y:0*
T0*
_output_shapes
: L
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_3Subsub_3/x:output:0truediv_2:z:0*
T0*
_output_shapes
: l
clip_by_value/MinimumMinimumtruediv_1:z:0	sub_3:z:0*
T0*'
_output_shapes
:��������� T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:��������� ]
mul_2Mul
Cast_1:y:0clip_by_value:z:0*
T0*'
_output_shapes
:��������� L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �:I
mul_3Mulmul_3/x:output:0Cast:y:0*
T0*
_output_shapes
: P
truediv_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
	truediv_3RealDivtruediv_3/x:output:0	mul_3:z:0*
T0*
_output_shapes
: L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �:K
mul_4Mul
Cast_1:y:0mul_4/y:output:0*
T0*
_output_shapes
: L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �:]
mul_5Multruediv:z:0mul_5/y:output:0*
T0*'
_output_shapes
:��������� I
Neg_1Neg	mul_5:z:0*
T0*'
_output_shapes
:��������� M
Round_1Round	mul_5:z:0*
T0*'
_output_shapes
:��������� X
add_2AddV2	Neg_1:y:0Round_1:y:0*
T0*'
_output_shapes
:��������� [
StopGradient_1StopGradient	add_2:z:0*
T0*'
_output_shapes
:��������� d
add_3AddV2	mul_5:z:0StopGradient_1:output:0*
T0*'
_output_shapes
:��������� X
mul_6Mul	add_3:z:0truediv_3:z:0*
T0*'
_output_shapes
:��������� ^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_value_1/MinimumMinimum	mul_6:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:��������� V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:��������� ^
mul_7Mul	mul_4:z:0clip_by_value_1:z:0*
T0*'
_output_shapes
:��������� V
add_4AddV2	mul_2:z:0	mul_7:z:0*
T0*'
_output_shapes
:��������� Q
Neg_2NegSelectV2:output:0*
T0*'
_output_shapes
:��������� V
add_5AddV2	Neg_2:y:0	add_4:z:0*
T0*'
_output_shapes
:��������� L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
mul_8Mulmul_8/x:output:0	add_5:z:0*
T0*'
_output_shapes
:��������� [
StopGradient_2StopGradient	mul_8:z:0*
T0*'
_output_shapes
:��������� l
add_6AddV2SelectV2:output:0StopGradient_2:output:0*
T0*'
_output_shapes
:��������� Q
IdentityIdentity	add_6:z:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
(__inference_dense1_layer_call_fn_8205188

inputs
unknown:, 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_8203214o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������,: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������,
 
_user_specified_nameinputs:'#
!
_user_specified_name	8205182:'#
!
_user_specified_name	8205184
�2
�
C__inference_z_mean_layer_call_and_return_conditional_losses_8204364

inputs)
readvariableop_resource:'
readvariableop_3_resource:
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   D[
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes

:N
truedivRealDivmul:z:0Cast:y:0*
T0*
_output_shapes

:@
NegNegtruediv:z:0*
T0*
_output_shapes

:D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:I
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:N
StopGradientStopGradientadd:z:0*
T0*
_output_shapes

:[
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ��Cv
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:R
mul_1MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

:P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   D^
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*
_output_shapes

:L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*
_output_shapes

:h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0O
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes

:M
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*
_output_shapes

:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*
_output_shapes

:R
StopGradient_1StopGradient	mul_3:z:0*
T0*
_output_shapes

:h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0j
add_3AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*
_output_shapes

:U
MatMulMatMulinputs	add_3:z:0*
T0*'
_output_shapes
:���������I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   D]
mul_4MulReadVariableOp_3:value:0mul_4/y:output:0*
T0*
_output_shapes
:P
	truediv_2RealDiv	mul_4:z:0
Cast_1:y:0*
T0*
_output_shapes
:@
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_4AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_4:z:0*
T0*
_output_shapes
:[
add_5AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ��Cv
clip_by_value_1/MinimumMinimum	add_5:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:R
mul_5Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   DZ
	truediv_3RealDiv	mul_5:z:0truediv_3/y:output:0*
T0*
_output_shapes
:L
mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_6Mulmul_6/x:output:0truediv_3:z:0*
T0*
_output_shapes
:f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0K
Neg_3NegReadVariableOp_4:value:0*
T0*
_output_shapes
:I
add_6AddV2	Neg_3:y:0	mul_6:z:0*
T0*
_output_shapes
:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_7Mulmul_7/x:output:0	add_6:z:0*
T0*
_output_shapes
:N
StopGradient_3StopGradient	mul_7:z:0*
T0*
_output_shapes
:f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0f
add_7AddV2ReadVariableOp_5:value:0StopGradient_3:output:0*
T0*
_output_shapes
:a
BiasAddBiasAddMatMul:product:0	add_7:z:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_5:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
9
inputs/
serving_default_inputs:0���������,:
z_mean0
StatefulPartitionedCall:0���������tensorflow/serving/predict:�
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
kernel_quantizer
bias_quantizer
kernel_quantizer_internal
bias_quantizer_internal

quantizers

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"beta_quantizer_internal
#gamma_quantizer_internal
$mean_quantizer_internal
%variance_quantizer_internal
&
quantizers
'axis
	(gamma
)beta
*moving_mean
+moving_variance"
_tf_keras_layer
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses
2
activation
2	quantizer"
_tf_keras_layer
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses
9kernel_quantizer
:bias_quantizer
9kernel_quantizer_internal
:bias_quantizer_internal
;
quantizers

<kernel
=bias"
_tf_keras_layer
�
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses
Dbeta_quantizer_internal
Egamma_quantizer_internal
Fmean_quantizer_internal
Gvariance_quantizer_internal
H
quantizers
Iaxis
	Jgamma
Kbeta
Lmoving_mean
Mmoving_variance"
_tf_keras_layer
�
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses
T
activation
T	quantizer"
_tf_keras_layer
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses
[kernel_quantizer
\bias_quantizer
[kernel_quantizer_internal
\bias_quantizer_internal
]
quantizers

^kernel
_bias"
_tf_keras_layer
�
0
1
(2
)3
*4
+5
<6
=7
J8
K9
L10
M11
^12
_13"
trackable_list_wrapper
f
0
1
(2
)3
<4
=5
J6
K7
^8
_9"
trackable_list_wrapper
 "
trackable_list_wrapper
�
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
etrace_0
ftrace_12�
4__inference_simplified_encoder_layer_call_fn_8205023
4__inference_simplified_encoder_layer_call_fn_8205056�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zetrace_0zftrace_1
�
gtrace_0
htrace_12�
O__inference_simplified_encoder_layer_call_and_return_conditional_losses_8204371
O__inference_simplified_encoder_layer_call_and_return_conditional_losses_8204990�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zgtrace_0zhtrace_1
�B�
"__inference__wrapped_model_8203144inputs"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
,
iserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
otrace_02�
(__inference_dense1_layer_call_fn_8205188�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zotrace_0
�
ptrace_02�
C__inference_dense1_layer_call_and_return_conditional_losses_8205256�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zptrace_0
"
_generic_user_object
"
_generic_user_object
.
0
1"
trackable_list_wrapper
:, 2dense1/kernel
: 2dense1/bias
<
(0
)1
*2
+3"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
�
vtrace_0
wtrace_12�
%__inference_BN1_layer_call_fn_8205269
%__inference_BN1_layer_call_fn_8205282�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zvtrace_0zwtrace_1
�
xtrace_0
ytrace_12�
@__inference_BN1_layer_call_and_return_conditional_losses_8205705
@__inference_BN1_layer_call_and_return_conditional_losses_8205995�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zxtrace_0zytrace_1
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
<
#0
"1
$2
%3"
trackable_list_wrapper
 "
trackable_list_wrapper
: 2	BN1/gamma
: 2BN1/beta
:  (2BN1/moving_mean
#:!  (2BN1/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
�
trace_02�
'__inference_relu1_layer_call_fn_8206000�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
�
�trace_02�
B__inference_relu1_layer_call_and_return_conditional_losses_8206069�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
"
_generic_user_object
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense2_layer_call_fn_8206078�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense2_layer_call_and_return_conditional_losses_8206146�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
"
_generic_user_object
"
_generic_user_object
.
90
:1"
trackable_list_wrapper
: 2dense2/kernel
:2dense2/bias
<
J0
K1
L2
M3"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
%__inference_BN2_layer_call_fn_8206159
%__inference_BN2_layer_call_fn_8206172�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
@__inference_BN2_layer_call_and_return_conditional_losses_8206595
@__inference_BN2_layer_call_and_return_conditional_losses_8206885�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
<
E0
D1
F2
G3"
trackable_list_wrapper
 "
trackable_list_wrapper
:2	BN2/gamma
:2BN2/beta
: (2BN2/moving_mean
#:! (2BN2/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_relu2_layer_call_fn_8206890�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_relu2_layer_call_and_return_conditional_losses_8206959�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
"
_generic_user_object
.
^0
_1"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_z_mean_layer_call_fn_8206968�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_z_mean_layer_call_and_return_conditional_losses_8207036�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
"
_generic_user_object
"
_generic_user_object
.
[0
\1"
trackable_list_wrapper
:2z_mean/kernel
:2z_mean/bias
<
*0
+1
L2
M3"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
4__inference_simplified_encoder_layer_call_fn_8205023inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
4__inference_simplified_encoder_layer_call_fn_8205056inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_simplified_encoder_layer_call_and_return_conditional_losses_8204371inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_simplified_encoder_layer_call_and_return_conditional_losses_8204990inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_signature_wrapper_8205179inputs"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense1_layer_call_fn_8205188inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense1_layer_call_and_return_conditional_losses_8205256inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_BN1_layer_call_fn_8205269inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_BN1_layer_call_fn_8205282inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_BN1_layer_call_and_return_conditional_losses_8205705inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_BN1_layer_call_and_return_conditional_losses_8205995inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_relu1_layer_call_fn_8206000inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_relu1_layer_call_and_return_conditional_losses_8206069inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense2_layer_call_fn_8206078inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense2_layer_call_and_return_conditional_losses_8206146inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_BN2_layer_call_fn_8206159inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_BN2_layer_call_fn_8206172inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_BN2_layer_call_and_return_conditional_losses_8206595inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_BN2_layer_call_and_return_conditional_losses_8206885inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_relu2_layer_call_fn_8206890inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_relu2_layer_call_and_return_conditional_losses_8206959inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_z_mean_layer_call_fn_8206968inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_z_mean_layer_call_and_return_conditional_losses_8207036inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
@__inference_BN1_layer_call_and_return_conditional_losses_8205705i()*+3�0
)�&
 �
inputs��������� 
p
� ",�)
"�
tensor_0��������� 
� �
@__inference_BN1_layer_call_and_return_conditional_losses_8205995i()*+3�0
)�&
 �
inputs��������� 
p 
� ",�)
"�
tensor_0��������� 
� �
%__inference_BN1_layer_call_fn_8205269^()*+3�0
)�&
 �
inputs��������� 
p
� "!�
unknown��������� �
%__inference_BN1_layer_call_fn_8205282^()*+3�0
)�&
 �
inputs��������� 
p 
� "!�
unknown��������� �
@__inference_BN2_layer_call_and_return_conditional_losses_8206595iJKLM3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
@__inference_BN2_layer_call_and_return_conditional_losses_8206885iJKLM3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
%__inference_BN2_layer_call_fn_8206159^JKLM3�0
)�&
 �
inputs���������
p
� "!�
unknown����������
%__inference_BN2_layer_call_fn_8206172^JKLM3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
"__inference__wrapped_model_8203144r()*+<=JKLM^_/�,
%�"
 �
inputs���������,
� "/�,
*
z_mean �
z_mean����������
C__inference_dense1_layer_call_and_return_conditional_losses_8205256c/�,
%�"
 �
inputs���������,
� ",�)
"�
tensor_0��������� 
� �
(__inference_dense1_layer_call_fn_8205188X/�,
%�"
 �
inputs���������,
� "!�
unknown��������� �
C__inference_dense2_layer_call_and_return_conditional_losses_8206146c<=/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0���������
� �
(__inference_dense2_layer_call_fn_8206078X<=/�,
%�"
 �
inputs��������� 
� "!�
unknown����������
B__inference_relu1_layer_call_and_return_conditional_losses_8206069_/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0��������� 
� 
'__inference_relu1_layer_call_fn_8206000T/�,
%�"
 �
inputs��������� 
� "!�
unknown��������� �
B__inference_relu2_layer_call_and_return_conditional_losses_8206959_/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� 
'__inference_relu2_layer_call_fn_8206890T/�,
%�"
 �
inputs���������
� "!�
unknown����������
%__inference_signature_wrapper_8205179|()*+<=JKLM^_9�6
� 
/�,
*
inputs �
inputs���������,"/�,
*
z_mean �
z_mean����������
O__inference_simplified_encoder_layer_call_and_return_conditional_losses_8204371w()*+<=JKLM^_7�4
-�*
 �
inputs���������,
p

 
� ",�)
"�
tensor_0���������
� �
O__inference_simplified_encoder_layer_call_and_return_conditional_losses_8204990w()*+<=JKLM^_7�4
-�*
 �
inputs���������,
p 

 
� ",�)
"�
tensor_0���������
� �
4__inference_simplified_encoder_layer_call_fn_8205023l()*+<=JKLM^_7�4
-�*
 �
inputs���������,
p

 
� "!�
unknown����������
4__inference_simplified_encoder_layer_call_fn_8205056l()*+<=JKLM^_7�4
-�*
 �
inputs���������,
p 

 
� "!�
unknown����������
C__inference_z_mean_layer_call_and_return_conditional_losses_8207036c^_/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
(__inference_z_mean_layer_call_fn_8206968X^_/�,
%�"
 �
inputs���������
� "!�
unknown���������