??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
3
Square
x"T
y"T"
Ttype:
2
	
?
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??
?
WordEmbedding_claims/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??2*0
shared_name!WordEmbedding_claims/embeddings
?
3WordEmbedding_claims/embeddings/Read/ReadVariableOpReadVariableOpWordEmbedding_claims/embeddings* 
_output_shapes
:
??2*
dtype0
?
!WordEmbedding_evidence/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??2*2
shared_name#!WordEmbedding_evidence/embeddings
?
5WordEmbedding_evidence/embeddings/Read/ReadVariableOpReadVariableOp!WordEmbedding_evidence/embeddings* 
_output_shapes
:
??2*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:f*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:f*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:f*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:f*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:f*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:f*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??

NoOpNoOp
?(
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*?'
value?'B?' B?'
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer_with_weights-2
layer-20
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
 
b

embeddings
	variables
trainable_variables
regularization_losses
 	keras_api
b
!
embeddings
"	variables
#trainable_variables
$regularization_losses
%	keras_api

&	keras_api

'	keras_api

(	keras_api

)	keras_api

*	keras_api

+	keras_api

,	keras_api

-	keras_api

.	keras_api

/	keras_api

0	keras_api

1	keras_api
R
2	variables
3trainable_variables
4regularization_losses
5	keras_api
R
6	variables
7trainable_variables
8regularization_losses
9	keras_api
R
:	variables
;trainable_variables
<regularization_losses
=	keras_api
R
>	variables
?trainable_variables
@regularization_losses
A	keras_api
h

Bkernel
Cbias
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
h
Hiter

Ibeta_1

Jbeta_2
	Kdecay
Llearning_rateBm?Cm?Bv?Cv?

0
!1
B2
C3

B0
C1
 
?
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
regularization_losses
 
om
VARIABLE_VALUEWordEmbedding_claims/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0
 
 
?
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
	variables
trainable_variables
regularization_losses
qo
VARIABLE_VALUE!WordEmbedding_evidence/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE

!0
 
 
?
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
"	variables
#trainable_variables
$regularization_losses
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
2	variables
3trainable_variables
4regularization_losses
 
 
 
?
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
6	variables
7trainable_variables
8regularization_losses
 
 
 
?
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
:	variables
;trainable_variables
<regularization_losses
 
 
 
?
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
>	variables
?trainable_variables
@regularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

B0
C1

B0
C1
 
?
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
D	variables
Etrainable_variables
Fregularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

0
!1
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20

u0
v1
 
 

0
 
 
 
 

!0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	wtotal
	xcount
y	variables
z	keras_api
D
	{total
	|count
}
_fn_kwargs
~	variables
	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

w0
x1

y	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

{0
|1

~	variables
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????`*
dtype0*
shape:?????????`
z
serving_default_input_2Placeholder*'
_output_shapes
:?????????`*
dtype0*
shape:?????????`
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2!WordEmbedding_evidence/embeddingsWordEmbedding_claims/embeddingsConstdense/kernel
dense/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_22027
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename3WordEmbedding_claims/embeddings/Read/ReadVariableOp5WordEmbedding_evidence/embeddings/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOpConst_1*
Tin
2	*
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
GPU 2J 8? *'
f"R 
__inference__traced_save_22362
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameWordEmbedding_claims/embeddings!WordEmbedding_evidence/embeddingsdense/kernel
dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/dense/kernel/mAdam/dense/bias/mAdam/dense/kernel/vAdam/dense/bias/v*
Tin
2*
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
GPU 2J 8? **
f%R#
!__inference__traced_restore_22423??
?
r
H__inference_concatenate_1_layer_call_and_return_conditional_losses_21717

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :u
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????3W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????3"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????2:?????????:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
#__inference_signature_wrapper_22027
input_1
input_2
unknown:
??2
	unknown_0:
??2
	unknown_1
	unknown_2:f
	unknown_3:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_21632o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:?????????`:?????????`: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????`
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????`
!
_user_specified_name	input_2:

_output_shapes
: 
?A
?
@__inference_model_layer_call_and_return_conditional_losses_22171
inputs_0
inputs_1A
-wordembedding_evidence_embedding_lookup_22120:
??2?
+wordembedding_claims_embedding_lookup_22126:
??2
tf_math_multiply_1_mul_y6
$dense_matmul_readvariableop_resource:f3
%dense_biasadd_readvariableop_resource:
identity??%WordEmbedding_claims/embedding_lookup?'WordEmbedding_evidence/embedding_lookup?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOpn
WordEmbedding_evidence/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:?????????`?
'WordEmbedding_evidence/embedding_lookupResourceGather-wordembedding_evidence_embedding_lookup_22120WordEmbedding_evidence/Cast:y:0*
Tindices0*@
_class6
42loc:@WordEmbedding_evidence/embedding_lookup/22120*+
_output_shapes
:?????????`2*
dtype0?
0WordEmbedding_evidence/embedding_lookup/IdentityIdentity0WordEmbedding_evidence/embedding_lookup:output:0*
T0*@
_class6
42loc:@WordEmbedding_evidence/embedding_lookup/22120*+
_output_shapes
:?????????`2?
2WordEmbedding_evidence/embedding_lookup/Identity_1Identity9WordEmbedding_evidence/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????`2l
WordEmbedding_claims/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:?????????`?
%WordEmbedding_claims/embedding_lookupResourceGather+wordembedding_claims_embedding_lookup_22126WordEmbedding_claims/Cast:y:0*
Tindices0*>
_class4
20loc:@WordEmbedding_claims/embedding_lookup/22126*+
_output_shapes
:?????????`2*
dtype0?
.WordEmbedding_claims/embedding_lookup/IdentityIdentity.WordEmbedding_claims/embedding_lookup:output:0*
T0*>
_class4
20loc:@WordEmbedding_claims/embedding_lookup/22126*+
_output_shapes
:?????????`2?
0WordEmbedding_claims/embedding_lookup/Identity_1Identity7WordEmbedding_claims/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????`2n
,tf.math.reduce_mean_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
tf.math.reduce_mean_1/MeanMean;WordEmbedding_evidence/embedding_lookup/Identity_1:output:05tf.math.reduce_mean_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2l
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
tf.math.reduce_mean/MeanMean9WordEmbedding_claims/embedding_lookup/Identity_1:output:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2?
(tf.math.l2_normalize/l2_normalize/SquareSquare!tf.math.reduce_mean/Mean:output:0*
T0*'
_output_shapes
:?????????2?
7tf.math.l2_normalize/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
%tf.math.l2_normalize/l2_normalize/SumSum,tf.math.l2_normalize/l2_normalize/Square:y:0@tf.math.l2_normalize/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(p
+tf.math.l2_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+?
)tf.math.l2_normalize/l2_normalize/MaximumMaximum.tf.math.l2_normalize/l2_normalize/Sum:output:04tf.math.l2_normalize/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:??????????
'tf.math.l2_normalize/l2_normalize/RsqrtRsqrt-tf.math.l2_normalize/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:??????????
!tf.math.l2_normalize/l2_normalizeMul!tf.math.reduce_mean/Mean:output:0+tf.math.l2_normalize/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:?????????2?
*tf.math.l2_normalize_1/l2_normalize/SquareSquare#tf.math.reduce_mean_1/Mean:output:0*
T0*'
_output_shapes
:?????????2?
9tf.math.l2_normalize_1/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
'tf.math.l2_normalize_1/l2_normalize/SumSum.tf.math.l2_normalize_1/l2_normalize/Square:y:0Btf.math.l2_normalize_1/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(r
-tf.math.l2_normalize_1/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+?
+tf.math.l2_normalize_1/l2_normalize/MaximumMaximum0tf.math.l2_normalize_1/l2_normalize/Sum:output:06tf.math.l2_normalize_1/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:??????????
)tf.math.l2_normalize_1/l2_normalize/RsqrtRsqrt/tf.math.l2_normalize_1/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:??????????
#tf.math.l2_normalize_1/l2_normalizeMul#tf.math.reduce_mean_1/Mean:output:0-tf.math.l2_normalize_1/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:?????????2?
tf.math.multiply/MulMul%tf.math.l2_normalize/l2_normalize:z:0'tf.math.l2_normalize_1/l2_normalize:z:0*
T0*'
_output_shapes
:?????????2s
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
tf.math.reduce_sum/SumSumtf.math.multiply/Mul:z:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????j
tf.math.negative/NegNegtf.math.reduce_sum/Sum:output:0*
T0*#
_output_shapes
:?????????
tf.math.multiply_1/MulMultf.math.negative/Neg:y:0tf_math_multiply_1_mul_y*
T0*#
_output_shapes
:?????????W
lambda/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
lambda/ExpandDims
ExpandDimstf.math.multiply_1/Mul:z:0lambda/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate/concatConcatV2!tf.math.reduce_mean/Mean:output:0lambda/ExpandDims:output:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????3[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_1/concatConcatV2#tf.math.reduce_mean_1/Mean:output:0lambda/ExpandDims:output:0"concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????3[
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_2/concatConcatV2concatenate/concat:output:0concatenate_1/concat:output:0"concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????f?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:f*
dtype0?
dense/MatMulMatMulconcatenate_2/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitydense/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp&^WordEmbedding_claims/embedding_lookup(^WordEmbedding_evidence/embedding_lookup^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:?????????`:?????????`: : : : : 2N
%WordEmbedding_claims/embedding_lookup%WordEmbedding_claims/embedding_lookup2R
'WordEmbedding_evidence/embedding_lookup'WordEmbedding_evidence/embedding_lookup2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:Q M
'
_output_shapes
:?????????`
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????`
"
_user_specified_name
inputs/1:

_output_shapes
: 
?
]
A__inference_lambda_layer_call_and_return_conditional_losses_22221

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :k

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????[
IdentityIdentityExpandDims:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:?????????:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
B
&__inference_lambda_layer_call_fn_22215

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_21804`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:?????????:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?F
?
 __inference__wrapped_model_21632
input_1
input_2G
3model_wordembedding_evidence_embedding_lookup_21581:
??2E
1model_wordembedding_claims_embedding_lookup_21587:
??2"
model_tf_math_multiply_1_mul_y<
*model_dense_matmul_readvariableop_resource:f9
+model_dense_biasadd_readvariableop_resource:
identity??+model/WordEmbedding_claims/embedding_lookup?-model/WordEmbedding_evidence/embedding_lookup?"model/dense/BiasAdd/ReadVariableOp?!model/dense/MatMul/ReadVariableOps
!model/WordEmbedding_evidence/CastCastinput_2*

DstT0*

SrcT0*'
_output_shapes
:?????????`?
-model/WordEmbedding_evidence/embedding_lookupResourceGather3model_wordembedding_evidence_embedding_lookup_21581%model/WordEmbedding_evidence/Cast:y:0*
Tindices0*F
_class<
:8loc:@model/WordEmbedding_evidence/embedding_lookup/21581*+
_output_shapes
:?????????`2*
dtype0?
6model/WordEmbedding_evidence/embedding_lookup/IdentityIdentity6model/WordEmbedding_evidence/embedding_lookup:output:0*
T0*F
_class<
:8loc:@model/WordEmbedding_evidence/embedding_lookup/21581*+
_output_shapes
:?????????`2?
8model/WordEmbedding_evidence/embedding_lookup/Identity_1Identity?model/WordEmbedding_evidence/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????`2q
model/WordEmbedding_claims/CastCastinput_1*

DstT0*

SrcT0*'
_output_shapes
:?????????`?
+model/WordEmbedding_claims/embedding_lookupResourceGather1model_wordembedding_claims_embedding_lookup_21587#model/WordEmbedding_claims/Cast:y:0*
Tindices0*D
_class:
86loc:@model/WordEmbedding_claims/embedding_lookup/21587*+
_output_shapes
:?????????`2*
dtype0?
4model/WordEmbedding_claims/embedding_lookup/IdentityIdentity4model/WordEmbedding_claims/embedding_lookup:output:0*
T0*D
_class:
86loc:@model/WordEmbedding_claims/embedding_lookup/21587*+
_output_shapes
:?????????`2?
6model/WordEmbedding_claims/embedding_lookup/Identity_1Identity=model/WordEmbedding_claims/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????`2t
2model/tf.math.reduce_mean_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
 model/tf.math.reduce_mean_1/MeanMeanAmodel/WordEmbedding_evidence/embedding_lookup/Identity_1:output:0;model/tf.math.reduce_mean_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2r
0model/tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
model/tf.math.reduce_mean/MeanMean?model/WordEmbedding_claims/embedding_lookup/Identity_1:output:09model/tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2?
.model/tf.math.l2_normalize/l2_normalize/SquareSquare'model/tf.math.reduce_mean/Mean:output:0*
T0*'
_output_shapes
:?????????2?
=model/tf.math.l2_normalize/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+model/tf.math.l2_normalize/l2_normalize/SumSum2model/tf.math.l2_normalize/l2_normalize/Square:y:0Fmodel/tf.math.l2_normalize/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(v
1model/tf.math.l2_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+?
/model/tf.math.l2_normalize/l2_normalize/MaximumMaximum4model/tf.math.l2_normalize/l2_normalize/Sum:output:0:model/tf.math.l2_normalize/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:??????????
-model/tf.math.l2_normalize/l2_normalize/RsqrtRsqrt3model/tf.math.l2_normalize/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:??????????
'model/tf.math.l2_normalize/l2_normalizeMul'model/tf.math.reduce_mean/Mean:output:01model/tf.math.l2_normalize/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:?????????2?
0model/tf.math.l2_normalize_1/l2_normalize/SquareSquare)model/tf.math.reduce_mean_1/Mean:output:0*
T0*'
_output_shapes
:?????????2?
?model/tf.math.l2_normalize_1/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
-model/tf.math.l2_normalize_1/l2_normalize/SumSum4model/tf.math.l2_normalize_1/l2_normalize/Square:y:0Hmodel/tf.math.l2_normalize_1/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(x
3model/tf.math.l2_normalize_1/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+?
1model/tf.math.l2_normalize_1/l2_normalize/MaximumMaximum6model/tf.math.l2_normalize_1/l2_normalize/Sum:output:0<model/tf.math.l2_normalize_1/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:??????????
/model/tf.math.l2_normalize_1/l2_normalize/RsqrtRsqrt5model/tf.math.l2_normalize_1/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:??????????
)model/tf.math.l2_normalize_1/l2_normalizeMul)model/tf.math.reduce_mean_1/Mean:output:03model/tf.math.l2_normalize_1/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:?????????2?
model/tf.math.multiply/MulMul+model/tf.math.l2_normalize/l2_normalize:z:0-model/tf.math.l2_normalize_1/l2_normalize:z:0*
T0*'
_output_shapes
:?????????2y
.model/tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
model/tf.math.reduce_sum/SumSummodel/tf.math.multiply/Mul:z:07model/tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????v
model/tf.math.negative/NegNeg%model/tf.math.reduce_sum/Sum:output:0*
T0*#
_output_shapes
:??????????
model/tf.math.multiply_1/MulMulmodel/tf.math.negative/Neg:y:0model_tf_math_multiply_1_mul_y*
T0*#
_output_shapes
:?????????]
model/lambda/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
model/lambda/ExpandDims
ExpandDims model/tf.math.multiply_1/Mul:z:0$model/lambda/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????_
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
model/concatenate/concatConcatV2'model/tf.math.reduce_mean/Mean:output:0 model/lambda/ExpandDims:output:0&model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????3a
model/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
model/concatenate_1/concatConcatV2)model/tf.math.reduce_mean_1/Mean:output:0 model/lambda/ExpandDims:output:0(model/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????3a
model/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
model/concatenate_2/concatConcatV2!model/concatenate/concat:output:0#model/concatenate_1/concat:output:0(model/concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????f?
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:f*
dtype0?
model/dense/MatMulMatMul#model/concatenate_2/concat:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????n
model/dense/SigmoidSigmoidmodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????f
IdentityIdentitymodel/dense/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^model/WordEmbedding_claims/embedding_lookup.^model/WordEmbedding_evidence/embedding_lookup#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:?????????`:?????????`: : : : : 2Z
+model/WordEmbedding_claims/embedding_lookup+model/WordEmbedding_claims/embedding_lookup2^
-model/WordEmbedding_evidence/embedding_lookup-model/WordEmbedding_evidence/embedding_lookup2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????`
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????`
!
_user_specified_name	input_2:

_output_shapes
: 
?
r
F__inference_concatenate_layer_call_and_return_conditional_losses_22240
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :w
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????3W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????3"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????2:?????????:Q M
'
_output_shapes
:?????????2
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?	
?
%__inference_model_layer_call_fn_22043
inputs_0
inputs_1
unknown:
??2
	unknown_0:
??2
	unknown_1
	unknown_2:f
	unknown_3:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_21746o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:?????????`:?????????`: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????`
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????`
"
_user_specified_name
inputs/1:

_output_shapes
: 
?
t
H__inference_concatenate_2_layer_call_and_return_conditional_losses_22266
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :w
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????fW
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????f"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????3:?????????3:Q M
'
_output_shapes
:?????????3
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????3
"
_user_specified_name
inputs/1
?	
?
%__inference_model_layer_call_fn_22059
inputs_0
inputs_1
unknown:
??2
	unknown_0:
??2
	unknown_1
	unknown_2:f
	unknown_3:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_21886o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:?????????`:?????????`: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????`
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????`
"
_user_specified_name
inputs/1:

_output_shapes
: 
?;
?
@__inference_model_layer_call_and_return_conditional_losses_22003
input_1
input_20
wordembedding_evidence_21963:
??2.
wordembedding_claims_21966:
??2
tf_math_multiply_1_mul_y
dense_21997:f
dense_21999:
identity??,WordEmbedding_claims/StatefulPartitionedCall?.WordEmbedding_evidence/StatefulPartitionedCall?dense/StatefulPartitionedCall?
.WordEmbedding_evidence/StatefulPartitionedCallStatefulPartitionedCallinput_2wordembedding_evidence_21963*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????`2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_WordEmbedding_evidence_layer_call_and_return_conditional_losses_21651?
,WordEmbedding_claims/StatefulPartitionedCallStatefulPartitionedCallinput_1wordembedding_claims_21966*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????`2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_WordEmbedding_claims_layer_call_and_return_conditional_losses_21665n
,tf.math.reduce_mean_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
tf.math.reduce_mean_1/MeanMean7WordEmbedding_evidence/StatefulPartitionedCall:output:05tf.math.reduce_mean_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2l
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
tf.math.reduce_mean/MeanMean5WordEmbedding_claims/StatefulPartitionedCall:output:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2?
(tf.math.l2_normalize/l2_normalize/SquareSquare!tf.math.reduce_mean/Mean:output:0*
T0*'
_output_shapes
:?????????2?
7tf.math.l2_normalize/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
%tf.math.l2_normalize/l2_normalize/SumSum,tf.math.l2_normalize/l2_normalize/Square:y:0@tf.math.l2_normalize/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(p
+tf.math.l2_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+?
)tf.math.l2_normalize/l2_normalize/MaximumMaximum.tf.math.l2_normalize/l2_normalize/Sum:output:04tf.math.l2_normalize/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:??????????
'tf.math.l2_normalize/l2_normalize/RsqrtRsqrt-tf.math.l2_normalize/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:??????????
!tf.math.l2_normalize/l2_normalizeMul!tf.math.reduce_mean/Mean:output:0+tf.math.l2_normalize/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:?????????2?
*tf.math.l2_normalize_1/l2_normalize/SquareSquare#tf.math.reduce_mean_1/Mean:output:0*
T0*'
_output_shapes
:?????????2?
9tf.math.l2_normalize_1/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
'tf.math.l2_normalize_1/l2_normalize/SumSum.tf.math.l2_normalize_1/l2_normalize/Square:y:0Btf.math.l2_normalize_1/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(r
-tf.math.l2_normalize_1/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+?
+tf.math.l2_normalize_1/l2_normalize/MaximumMaximum0tf.math.l2_normalize_1/l2_normalize/Sum:output:06tf.math.l2_normalize_1/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:??????????
)tf.math.l2_normalize_1/l2_normalize/RsqrtRsqrt/tf.math.l2_normalize_1/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:??????????
#tf.math.l2_normalize_1/l2_normalizeMul#tf.math.reduce_mean_1/Mean:output:0-tf.math.l2_normalize_1/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:?????????2?
tf.math.multiply/MulMul%tf.math.l2_normalize/l2_normalize:z:0'tf.math.l2_normalize_1/l2_normalize:z:0*
T0*'
_output_shapes
:?????????2s
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
tf.math.reduce_sum/SumSumtf.math.multiply/Mul:z:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????j
tf.math.negative/NegNegtf.math.reduce_sum/Sum:output:0*
T0*#
_output_shapes
:?????????
tf.math.multiply_1/MulMultf.math.negative/Neg:y:0tf_math_multiply_1_mul_y*
T0*#
_output_shapes
:??????????
lambda/PartitionedCallPartitionedCalltf.math.multiply_1/Mul:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_21804?
concatenate/PartitionedCallPartitionedCall!tf.math.reduce_mean/Mean:output:0lambda/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_21708?
concatenate_1/PartitionedCallPartitionedCall#tf.math.reduce_mean_1/Mean:output:0lambda/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_21717?
concatenate_2/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0&concatenate_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????f* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_21726?
dense/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_21997dense_21999*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_21739u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp-^WordEmbedding_claims/StatefulPartitionedCall/^WordEmbedding_evidence/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:?????????`:?????????`: : : : : 2\
,WordEmbedding_claims/StatefulPartitionedCall,WordEmbedding_claims/StatefulPartitionedCall2`
.WordEmbedding_evidence/StatefulPartitionedCall.WordEmbedding_evidence/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:P L
'
_output_shapes
:?????????`
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????`
!
_user_specified_name	input_2:

_output_shapes
: 
?
Y
-__inference_concatenate_2_layer_call_fn_22259
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????f* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_21726`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????f"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????3:?????????3:Q M
'
_output_shapes
:?????????3
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????3
"
_user_specified_name
inputs/1
?	
?
O__inference_WordEmbedding_claims_layer_call_and_return_conditional_losses_21665

inputs*
embedding_lookup_21659:
??2
identity??embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????`?
embedding_lookupResourceGatherembedding_lookup_21659Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/21659*+
_output_shapes
:?????????`2*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/21659*+
_output_shapes
:?????????`2?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????`2w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????`2Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????`: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
p
F__inference_concatenate_layer_call_and_return_conditional_losses_21708

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :u
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????3W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????3"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????2:?????????:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
6__inference_WordEmbedding_evidence_layer_call_fn_22195

inputs
unknown:
??2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????`2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_WordEmbedding_evidence_layer_call_and_return_conditional_losses_21651s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????`: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????`
 
_user_specified_nameinputs
?;
?
@__inference_model_layer_call_and_return_conditional_losses_21886

inputs
inputs_10
wordembedding_evidence_21846:
??2.
wordembedding_claims_21849:
??2
tf_math_multiply_1_mul_y
dense_21880:f
dense_21882:
identity??,WordEmbedding_claims/StatefulPartitionedCall?.WordEmbedding_evidence/StatefulPartitionedCall?dense/StatefulPartitionedCall?
.WordEmbedding_evidence/StatefulPartitionedCallStatefulPartitionedCallinputs_1wordembedding_evidence_21846*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????`2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_WordEmbedding_evidence_layer_call_and_return_conditional_losses_21651?
,WordEmbedding_claims/StatefulPartitionedCallStatefulPartitionedCallinputswordembedding_claims_21849*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????`2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_WordEmbedding_claims_layer_call_and_return_conditional_losses_21665n
,tf.math.reduce_mean_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
tf.math.reduce_mean_1/MeanMean7WordEmbedding_evidence/StatefulPartitionedCall:output:05tf.math.reduce_mean_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2l
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
tf.math.reduce_mean/MeanMean5WordEmbedding_claims/StatefulPartitionedCall:output:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2?
(tf.math.l2_normalize/l2_normalize/SquareSquare!tf.math.reduce_mean/Mean:output:0*
T0*'
_output_shapes
:?????????2?
7tf.math.l2_normalize/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
%tf.math.l2_normalize/l2_normalize/SumSum,tf.math.l2_normalize/l2_normalize/Square:y:0@tf.math.l2_normalize/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(p
+tf.math.l2_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+?
)tf.math.l2_normalize/l2_normalize/MaximumMaximum.tf.math.l2_normalize/l2_normalize/Sum:output:04tf.math.l2_normalize/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:??????????
'tf.math.l2_normalize/l2_normalize/RsqrtRsqrt-tf.math.l2_normalize/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:??????????
!tf.math.l2_normalize/l2_normalizeMul!tf.math.reduce_mean/Mean:output:0+tf.math.l2_normalize/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:?????????2?
*tf.math.l2_normalize_1/l2_normalize/SquareSquare#tf.math.reduce_mean_1/Mean:output:0*
T0*'
_output_shapes
:?????????2?
9tf.math.l2_normalize_1/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
'tf.math.l2_normalize_1/l2_normalize/SumSum.tf.math.l2_normalize_1/l2_normalize/Square:y:0Btf.math.l2_normalize_1/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(r
-tf.math.l2_normalize_1/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+?
+tf.math.l2_normalize_1/l2_normalize/MaximumMaximum0tf.math.l2_normalize_1/l2_normalize/Sum:output:06tf.math.l2_normalize_1/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:??????????
)tf.math.l2_normalize_1/l2_normalize/RsqrtRsqrt/tf.math.l2_normalize_1/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:??????????
#tf.math.l2_normalize_1/l2_normalizeMul#tf.math.reduce_mean_1/Mean:output:0-tf.math.l2_normalize_1/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:?????????2?
tf.math.multiply/MulMul%tf.math.l2_normalize/l2_normalize:z:0'tf.math.l2_normalize_1/l2_normalize:z:0*
T0*'
_output_shapes
:?????????2s
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
tf.math.reduce_sum/SumSumtf.math.multiply/Mul:z:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????j
tf.math.negative/NegNegtf.math.reduce_sum/Sum:output:0*
T0*#
_output_shapes
:?????????
tf.math.multiply_1/MulMultf.math.negative/Neg:y:0tf_math_multiply_1_mul_y*
T0*#
_output_shapes
:??????????
lambda/PartitionedCallPartitionedCalltf.math.multiply_1/Mul:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_21804?
concatenate/PartitionedCallPartitionedCall!tf.math.reduce_mean/Mean:output:0lambda/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_21708?
concatenate_1/PartitionedCallPartitionedCall#tf.math.reduce_mean_1/Mean:output:0lambda/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_21717?
concatenate_2/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0&concatenate_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????f* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_21726?
dense/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_21880dense_21882*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_21739u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp-^WordEmbedding_claims/StatefulPartitionedCall/^WordEmbedding_evidence/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:?????????`:?????????`: : : : : 2\
,WordEmbedding_claims/StatefulPartitionedCall,WordEmbedding_claims/StatefulPartitionedCall2`
.WordEmbedding_evidence/StatefulPartitionedCall.WordEmbedding_evidence/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
'
_output_shapes
:?????????`
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????`
 
_user_specified_nameinputs:

_output_shapes
: 
?
]
A__inference_lambda_layer_call_and_return_conditional_losses_21699

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :k

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????[
IdentityIdentityExpandDims:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:?????????:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_dense_layer_call_fn_22275

inputs
unknown:f
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_21739o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????f: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????f
 
_user_specified_nameinputs
?	
?
%__inference_model_layer_call_fn_21915
input_1
input_2
unknown:
??2
	unknown_0:
??2
	unknown_1
	unknown_2:f
	unknown_3:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_21886o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:?????????`:?????????`: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????`
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????`
!
_user_specified_name	input_2:

_output_shapes
: 
?A
?
@__inference_model_layer_call_and_return_conditional_losses_22115
inputs_0
inputs_1A
-wordembedding_evidence_embedding_lookup_22064:
??2?
+wordembedding_claims_embedding_lookup_22070:
??2
tf_math_multiply_1_mul_y6
$dense_matmul_readvariableop_resource:f3
%dense_biasadd_readvariableop_resource:
identity??%WordEmbedding_claims/embedding_lookup?'WordEmbedding_evidence/embedding_lookup?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOpn
WordEmbedding_evidence/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:?????????`?
'WordEmbedding_evidence/embedding_lookupResourceGather-wordembedding_evidence_embedding_lookup_22064WordEmbedding_evidence/Cast:y:0*
Tindices0*@
_class6
42loc:@WordEmbedding_evidence/embedding_lookup/22064*+
_output_shapes
:?????????`2*
dtype0?
0WordEmbedding_evidence/embedding_lookup/IdentityIdentity0WordEmbedding_evidence/embedding_lookup:output:0*
T0*@
_class6
42loc:@WordEmbedding_evidence/embedding_lookup/22064*+
_output_shapes
:?????????`2?
2WordEmbedding_evidence/embedding_lookup/Identity_1Identity9WordEmbedding_evidence/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????`2l
WordEmbedding_claims/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:?????????`?
%WordEmbedding_claims/embedding_lookupResourceGather+wordembedding_claims_embedding_lookup_22070WordEmbedding_claims/Cast:y:0*
Tindices0*>
_class4
20loc:@WordEmbedding_claims/embedding_lookup/22070*+
_output_shapes
:?????????`2*
dtype0?
.WordEmbedding_claims/embedding_lookup/IdentityIdentity.WordEmbedding_claims/embedding_lookup:output:0*
T0*>
_class4
20loc:@WordEmbedding_claims/embedding_lookup/22070*+
_output_shapes
:?????????`2?
0WordEmbedding_claims/embedding_lookup/Identity_1Identity7WordEmbedding_claims/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????`2n
,tf.math.reduce_mean_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
tf.math.reduce_mean_1/MeanMean;WordEmbedding_evidence/embedding_lookup/Identity_1:output:05tf.math.reduce_mean_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2l
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
tf.math.reduce_mean/MeanMean9WordEmbedding_claims/embedding_lookup/Identity_1:output:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2?
(tf.math.l2_normalize/l2_normalize/SquareSquare!tf.math.reduce_mean/Mean:output:0*
T0*'
_output_shapes
:?????????2?
7tf.math.l2_normalize/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
%tf.math.l2_normalize/l2_normalize/SumSum,tf.math.l2_normalize/l2_normalize/Square:y:0@tf.math.l2_normalize/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(p
+tf.math.l2_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+?
)tf.math.l2_normalize/l2_normalize/MaximumMaximum.tf.math.l2_normalize/l2_normalize/Sum:output:04tf.math.l2_normalize/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:??????????
'tf.math.l2_normalize/l2_normalize/RsqrtRsqrt-tf.math.l2_normalize/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:??????????
!tf.math.l2_normalize/l2_normalizeMul!tf.math.reduce_mean/Mean:output:0+tf.math.l2_normalize/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:?????????2?
*tf.math.l2_normalize_1/l2_normalize/SquareSquare#tf.math.reduce_mean_1/Mean:output:0*
T0*'
_output_shapes
:?????????2?
9tf.math.l2_normalize_1/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
'tf.math.l2_normalize_1/l2_normalize/SumSum.tf.math.l2_normalize_1/l2_normalize/Square:y:0Btf.math.l2_normalize_1/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(r
-tf.math.l2_normalize_1/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+?
+tf.math.l2_normalize_1/l2_normalize/MaximumMaximum0tf.math.l2_normalize_1/l2_normalize/Sum:output:06tf.math.l2_normalize_1/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:??????????
)tf.math.l2_normalize_1/l2_normalize/RsqrtRsqrt/tf.math.l2_normalize_1/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:??????????
#tf.math.l2_normalize_1/l2_normalizeMul#tf.math.reduce_mean_1/Mean:output:0-tf.math.l2_normalize_1/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:?????????2?
tf.math.multiply/MulMul%tf.math.l2_normalize/l2_normalize:z:0'tf.math.l2_normalize_1/l2_normalize:z:0*
T0*'
_output_shapes
:?????????2s
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
tf.math.reduce_sum/SumSumtf.math.multiply/Mul:z:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????j
tf.math.negative/NegNegtf.math.reduce_sum/Sum:output:0*
T0*#
_output_shapes
:?????????
tf.math.multiply_1/MulMultf.math.negative/Neg:y:0tf_math_multiply_1_mul_y*
T0*#
_output_shapes
:?????????W
lambda/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
lambda/ExpandDims
ExpandDimstf.math.multiply_1/Mul:z:0lambda/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate/concatConcatV2!tf.math.reduce_mean/Mean:output:0lambda/ExpandDims:output:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????3[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_1/concatConcatV2#tf.math.reduce_mean_1/Mean:output:0lambda/ExpandDims:output:0"concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????3[
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_2/concatConcatV2concatenate/concat:output:0concatenate_1/concat:output:0"concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????f?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:f*
dtype0?
dense/MatMulMatMulconcatenate_2/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitydense/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp&^WordEmbedding_claims/embedding_lookup(^WordEmbedding_evidence/embedding_lookup^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:?????????`:?????????`: : : : : 2N
%WordEmbedding_claims/embedding_lookup%WordEmbedding_claims/embedding_lookup2R
'WordEmbedding_evidence/embedding_lookup'WordEmbedding_evidence/embedding_lookup2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:Q M
'
_output_shapes
:?????????`
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????`
"
_user_specified_name
inputs/1:

_output_shapes
: 
?
?
4__inference_WordEmbedding_claims_layer_call_fn_22178

inputs
unknown:
??2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????`2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_WordEmbedding_claims_layer_call_and_return_conditional_losses_21665s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????`: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
W
+__inference_concatenate_layer_call_fn_22233
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_21708`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????3"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????2:?????????:Q M
'
_output_shapes
:?????????2
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?

?
@__inference_dense_layer_call_and_return_conditional_losses_22286

inputs0
matmul_readvariableop_resource:f-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:f*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????f: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????f
 
_user_specified_nameinputs
?

?
@__inference_dense_layer_call_and_return_conditional_losses_21739

inputs0
matmul_readvariableop_resource:f-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:f*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????f: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????f
 
_user_specified_nameinputs
?	
?
O__inference_WordEmbedding_claims_layer_call_and_return_conditional_losses_22188

inputs*
embedding_lookup_22182:
??2
identity??embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????`?
embedding_lookupResourceGatherembedding_lookup_22182Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/22182*+
_output_shapes
:?????????`2*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/22182*+
_output_shapes
:?????????`2?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????`2w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????`2Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????`: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????`
 
_user_specified_nameinputs
?*
?
__inference__traced_save_22362
file_prefix>
:savev2_wordembedding_claims_embeddings_read_readvariableop@
<savev2_wordembedding_evidence_embeddings_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop
savev2_const_1

identity_1??MergeV2Checkpointsw
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
_temp/part?
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
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0:savev2_wordembedding_claims_embeddings_read_readvariableop<savev2_wordembedding_evidence_embeddings_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableopsavev2_const_1"/device:CPU:0*
_output_shapes
 * 
dtypes
2	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*q
_input_shapes`
^: :
??2:
??2:f:: : : : : : : : : :f::f:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??2:&"
 
_output_shapes
:
??2:$ 

_output_shapes

:f: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:f: 

_output_shapes
::$ 

_output_shapes

:f: 

_output_shapes
::

_output_shapes
: 
?
]
A__inference_lambda_layer_call_and_return_conditional_losses_22227

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :k

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????[
IdentityIdentityExpandDims:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:?????????:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
]
A__inference_lambda_layer_call_and_return_conditional_losses_21804

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :k

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????[
IdentityIdentityExpandDims:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:?????????:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
t
H__inference_concatenate_1_layer_call_and_return_conditional_losses_22253
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :w
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????3W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????3"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????2:?????????:Q M
'
_output_shapes
:?????????2
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?	
?
Q__inference_WordEmbedding_evidence_layer_call_and_return_conditional_losses_21651

inputs*
embedding_lookup_21645:
??2
identity??embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????`?
embedding_lookupResourceGatherembedding_lookup_21645Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/21645*+
_output_shapes
:?????????`2*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/21645*+
_output_shapes
:?????????`2?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????`2w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????`2Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????`: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????`
 
_user_specified_nameinputs
?E
?	
!__inference__traced_restore_22423
file_prefixD
0assignvariableop_wordembedding_claims_embeddings:
??2H
4assignvariableop_1_wordembedding_evidence_embeddings:
??21
assignvariableop_2_dense_kernel:f+
assignvariableop_3_dense_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: "
assignvariableop_9_total: #
assignvariableop_10_count: %
assignvariableop_11_total_1: %
assignvariableop_12_count_1: 9
'assignvariableop_13_adam_dense_kernel_m:f3
%assignvariableop_14_adam_dense_bias_m:9
'assignvariableop_15_adam_dense_kernel_v:f3
%assignvariableop_16_adam_dense_bias_v:
identity_18??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*\
_output_shapesJ
H::::::::::::::::::* 
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp0assignvariableop_wordembedding_claims_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp4assignvariableop_1_wordembedding_evidence_embeddingsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp'assignvariableop_13_adam_dense_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp%assignvariableop_14_adam_dense_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp'assignvariableop_15_adam_dense_kernel_vIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp%assignvariableop_16_adam_dense_bias_vIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_17Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_18IdentityIdentity_17:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_18Identity_18:output:0*7
_input_shapes&
$: : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162(
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
_user_specified_namefile_prefix
?
B
&__inference_lambda_layer_call_fn_22210

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_21699`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:?????????:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?;
?
@__inference_model_layer_call_and_return_conditional_losses_21746

inputs
inputs_10
wordembedding_evidence_21652:
??2.
wordembedding_claims_21666:
??2
tf_math_multiply_1_mul_y
dense_21740:f
dense_21742:
identity??,WordEmbedding_claims/StatefulPartitionedCall?.WordEmbedding_evidence/StatefulPartitionedCall?dense/StatefulPartitionedCall?
.WordEmbedding_evidence/StatefulPartitionedCallStatefulPartitionedCallinputs_1wordembedding_evidence_21652*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????`2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_WordEmbedding_evidence_layer_call_and_return_conditional_losses_21651?
,WordEmbedding_claims/StatefulPartitionedCallStatefulPartitionedCallinputswordembedding_claims_21666*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????`2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_WordEmbedding_claims_layer_call_and_return_conditional_losses_21665n
,tf.math.reduce_mean_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
tf.math.reduce_mean_1/MeanMean7WordEmbedding_evidence/StatefulPartitionedCall:output:05tf.math.reduce_mean_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2l
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
tf.math.reduce_mean/MeanMean5WordEmbedding_claims/StatefulPartitionedCall:output:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2?
(tf.math.l2_normalize/l2_normalize/SquareSquare!tf.math.reduce_mean/Mean:output:0*
T0*'
_output_shapes
:?????????2?
7tf.math.l2_normalize/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
%tf.math.l2_normalize/l2_normalize/SumSum,tf.math.l2_normalize/l2_normalize/Square:y:0@tf.math.l2_normalize/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(p
+tf.math.l2_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+?
)tf.math.l2_normalize/l2_normalize/MaximumMaximum.tf.math.l2_normalize/l2_normalize/Sum:output:04tf.math.l2_normalize/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:??????????
'tf.math.l2_normalize/l2_normalize/RsqrtRsqrt-tf.math.l2_normalize/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:??????????
!tf.math.l2_normalize/l2_normalizeMul!tf.math.reduce_mean/Mean:output:0+tf.math.l2_normalize/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:?????????2?
*tf.math.l2_normalize_1/l2_normalize/SquareSquare#tf.math.reduce_mean_1/Mean:output:0*
T0*'
_output_shapes
:?????????2?
9tf.math.l2_normalize_1/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
'tf.math.l2_normalize_1/l2_normalize/SumSum.tf.math.l2_normalize_1/l2_normalize/Square:y:0Btf.math.l2_normalize_1/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(r
-tf.math.l2_normalize_1/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+?
+tf.math.l2_normalize_1/l2_normalize/MaximumMaximum0tf.math.l2_normalize_1/l2_normalize/Sum:output:06tf.math.l2_normalize_1/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:??????????
)tf.math.l2_normalize_1/l2_normalize/RsqrtRsqrt/tf.math.l2_normalize_1/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:??????????
#tf.math.l2_normalize_1/l2_normalizeMul#tf.math.reduce_mean_1/Mean:output:0-tf.math.l2_normalize_1/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:?????????2?
tf.math.multiply/MulMul%tf.math.l2_normalize/l2_normalize:z:0'tf.math.l2_normalize_1/l2_normalize:z:0*
T0*'
_output_shapes
:?????????2s
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
tf.math.reduce_sum/SumSumtf.math.multiply/Mul:z:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????j
tf.math.negative/NegNegtf.math.reduce_sum/Sum:output:0*
T0*#
_output_shapes
:?????????
tf.math.multiply_1/MulMultf.math.negative/Neg:y:0tf_math_multiply_1_mul_y*
T0*#
_output_shapes
:??????????
lambda/PartitionedCallPartitionedCalltf.math.multiply_1/Mul:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_21699?
concatenate/PartitionedCallPartitionedCall!tf.math.reduce_mean/Mean:output:0lambda/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_21708?
concatenate_1/PartitionedCallPartitionedCall#tf.math.reduce_mean_1/Mean:output:0lambda/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_21717?
concatenate_2/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0&concatenate_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????f* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_21726?
dense/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_21740dense_21742*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_21739u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp-^WordEmbedding_claims/StatefulPartitionedCall/^WordEmbedding_evidence/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:?????????`:?????????`: : : : : 2\
,WordEmbedding_claims/StatefulPartitionedCall,WordEmbedding_claims/StatefulPartitionedCall2`
.WordEmbedding_evidence/StatefulPartitionedCall.WordEmbedding_evidence/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
'
_output_shapes
:?????????`
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????`
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
%__inference_model_layer_call_fn_21759
input_1
input_2
unknown:
??2
	unknown_0:
??2
	unknown_1
	unknown_2:f
	unknown_3:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_21746o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:?????????`:?????????`: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????`
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????`
!
_user_specified_name	input_2:

_output_shapes
: 
?
Y
-__inference_concatenate_1_layer_call_fn_22246
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_21717`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????3"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????2:?????????:Q M
'
_output_shapes
:?????????2
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?	
?
Q__inference_WordEmbedding_evidence_layer_call_and_return_conditional_losses_22205

inputs*
embedding_lookup_22199:
??2
identity??embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????`?
embedding_lookupResourceGatherembedding_lookup_22199Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/22199*+
_output_shapes
:?????????`2*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/22199*+
_output_shapes
:?????????`2?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????`2w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????`2Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????`: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
r
H__inference_concatenate_2_layer_call_and_return_conditional_losses_21726

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :u
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????fW
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????f"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????3:?????????3:O K
'
_output_shapes
:?????????3
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????3
 
_user_specified_nameinputs
?;
?
@__inference_model_layer_call_and_return_conditional_losses_21959
input_1
input_20
wordembedding_evidence_21919:
??2.
wordembedding_claims_21922:
??2
tf_math_multiply_1_mul_y
dense_21953:f
dense_21955:
identity??,WordEmbedding_claims/StatefulPartitionedCall?.WordEmbedding_evidence/StatefulPartitionedCall?dense/StatefulPartitionedCall?
.WordEmbedding_evidence/StatefulPartitionedCallStatefulPartitionedCallinput_2wordembedding_evidence_21919*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????`2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_WordEmbedding_evidence_layer_call_and_return_conditional_losses_21651?
,WordEmbedding_claims/StatefulPartitionedCallStatefulPartitionedCallinput_1wordembedding_claims_21922*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????`2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_WordEmbedding_claims_layer_call_and_return_conditional_losses_21665n
,tf.math.reduce_mean_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
tf.math.reduce_mean_1/MeanMean7WordEmbedding_evidence/StatefulPartitionedCall:output:05tf.math.reduce_mean_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2l
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
tf.math.reduce_mean/MeanMean5WordEmbedding_claims/StatefulPartitionedCall:output:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????2?
(tf.math.l2_normalize/l2_normalize/SquareSquare!tf.math.reduce_mean/Mean:output:0*
T0*'
_output_shapes
:?????????2?
7tf.math.l2_normalize/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
%tf.math.l2_normalize/l2_normalize/SumSum,tf.math.l2_normalize/l2_normalize/Square:y:0@tf.math.l2_normalize/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(p
+tf.math.l2_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+?
)tf.math.l2_normalize/l2_normalize/MaximumMaximum.tf.math.l2_normalize/l2_normalize/Sum:output:04tf.math.l2_normalize/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:??????????
'tf.math.l2_normalize/l2_normalize/RsqrtRsqrt-tf.math.l2_normalize/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:??????????
!tf.math.l2_normalize/l2_normalizeMul!tf.math.reduce_mean/Mean:output:0+tf.math.l2_normalize/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:?????????2?
*tf.math.l2_normalize_1/l2_normalize/SquareSquare#tf.math.reduce_mean_1/Mean:output:0*
T0*'
_output_shapes
:?????????2?
9tf.math.l2_normalize_1/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
'tf.math.l2_normalize_1/l2_normalize/SumSum.tf.math.l2_normalize_1/l2_normalize/Square:y:0Btf.math.l2_normalize_1/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(r
-tf.math.l2_normalize_1/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+?
+tf.math.l2_normalize_1/l2_normalize/MaximumMaximum0tf.math.l2_normalize_1/l2_normalize/Sum:output:06tf.math.l2_normalize_1/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:??????????
)tf.math.l2_normalize_1/l2_normalize/RsqrtRsqrt/tf.math.l2_normalize_1/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:??????????
#tf.math.l2_normalize_1/l2_normalizeMul#tf.math.reduce_mean_1/Mean:output:0-tf.math.l2_normalize_1/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:?????????2?
tf.math.multiply/MulMul%tf.math.l2_normalize/l2_normalize:z:0'tf.math.l2_normalize_1/l2_normalize:z:0*
T0*'
_output_shapes
:?????????2s
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
tf.math.reduce_sum/SumSumtf.math.multiply/Mul:z:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????j
tf.math.negative/NegNegtf.math.reduce_sum/Sum:output:0*
T0*#
_output_shapes
:?????????
tf.math.multiply_1/MulMultf.math.negative/Neg:y:0tf_math_multiply_1_mul_y*
T0*#
_output_shapes
:??????????
lambda/PartitionedCallPartitionedCalltf.math.multiply_1/Mul:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_21699?
concatenate/PartitionedCallPartitionedCall!tf.math.reduce_mean/Mean:output:0lambda/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_21708?
concatenate_1/PartitionedCallPartitionedCall#tf.math.reduce_mean_1/Mean:output:0lambda/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_21717?
concatenate_2/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0&concatenate_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????f* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_21726?
dense/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_21953dense_21955*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_21739u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp-^WordEmbedding_claims/StatefulPartitionedCall/^WordEmbedding_evidence/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:?????????`:?????????`: : : : : 2\
,WordEmbedding_claims/StatefulPartitionedCall,WordEmbedding_claims/StatefulPartitionedCall2`
.WordEmbedding_evidence/StatefulPartitionedCall.WordEmbedding_evidence/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:P L
'
_output_shapes
:?????????`
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????`
!
_user_specified_name	input_2:

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????`
;
input_20
serving_default_input_2:0?????????`9
dense0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer_with_weights-2
layer-20
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
?

embeddings
	variables
trainable_variables
regularization_losses
 	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
!
embeddings
"	variables
#trainable_variables
$regularization_losses
%	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
(
&	keras_api"
_tf_keras_layer
(
'	keras_api"
_tf_keras_layer
(
(	keras_api"
_tf_keras_layer
(
)	keras_api"
_tf_keras_layer
(
*	keras_api"
_tf_keras_layer
(
+	keras_api"
_tf_keras_layer
(
,	keras_api"
_tf_keras_layer
(
-	keras_api"
_tf_keras_layer
(
.	keras_api"
_tf_keras_layer
(
/	keras_api"
_tf_keras_layer
(
0	keras_api"
_tf_keras_layer
(
1	keras_api"
_tf_keras_layer
?
2	variables
3trainable_variables
4regularization_losses
5	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
6	variables
7trainable_variables
8regularization_losses
9	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
:	variables
;trainable_variables
<regularization_losses
=	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
>	variables
?trainable_variables
@regularization_losses
A	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Bkernel
Cbias
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
{
Hiter

Ibeta_1

Jbeta_2
	Kdecay
Llearning_rateBm?Cm?Bv?Cv?"
	optimizer
<
0
!1
B2
C3"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
3:1
??22WordEmbedding_claims/embeddings
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
5:3
??22!WordEmbedding_evidence/embeddings
'
!0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
"	variables
#trainable_variables
$regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
2	variables
3trainable_variables
4regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
6	variables
7trainable_variables
8regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
:	variables
;trainable_variables
<regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
>	variables
?trainable_variables
@regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:f2dense/kernel
:2
dense/bias
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
D	variables
Etrainable_variables
Fregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.
0
!1"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20"
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
!0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
N
	wtotal
	xcount
y	variables
z	keras_api"
_tf_keras_metric
^
	{total
	|count
}
_fn_kwargs
~	variables
	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
w0
x1"
trackable_list_wrapper
-
y	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
{0
|1"
trackable_list_wrapper
-
~	variables"
_generic_user_object
#:!f2Adam/dense/kernel/m
:2Adam/dense/bias/m
#:!f2Adam/dense/kernel/v
:2Adam/dense/bias/v
?2?
%__inference_model_layer_call_fn_21759
%__inference_model_layer_call_fn_22043
%__inference_model_layer_call_fn_22059
%__inference_model_layer_call_fn_21915?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
@__inference_model_layer_call_and_return_conditional_losses_22115
@__inference_model_layer_call_and_return_conditional_losses_22171
@__inference_model_layer_call_and_return_conditional_losses_21959
@__inference_model_layer_call_and_return_conditional_losses_22003?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
 __inference__wrapped_model_21632input_1input_2"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_WordEmbedding_claims_layer_call_fn_22178?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_WordEmbedding_claims_layer_call_and_return_conditional_losses_22188?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
6__inference_WordEmbedding_evidence_layer_call_fn_22195?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Q__inference_WordEmbedding_evidence_layer_call_and_return_conditional_losses_22205?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_lambda_layer_call_fn_22210
&__inference_lambda_layer_call_fn_22215?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_lambda_layer_call_and_return_conditional_losses_22221
A__inference_lambda_layer_call_and_return_conditional_losses_22227?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_concatenate_layer_call_fn_22233?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_concatenate_layer_call_and_return_conditional_losses_22240?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_concatenate_1_layer_call_fn_22246?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_concatenate_1_layer_call_and_return_conditional_losses_22253?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_concatenate_2_layer_call_fn_22259?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_concatenate_2_layer_call_and_return_conditional_losses_22266?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_dense_layer_call_fn_22275?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense_layer_call_and_return_conditional_losses_22286?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_22027input_1input_2"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
	J
Const?
O__inference_WordEmbedding_claims_layer_call_and_return_conditional_losses_22188_/?,
%?"
 ?
inputs?????????`
? ")?&
?
0?????????`2
? ?
4__inference_WordEmbedding_claims_layer_call_fn_22178R/?,
%?"
 ?
inputs?????????`
? "??????????`2?
Q__inference_WordEmbedding_evidence_layer_call_and_return_conditional_losses_22205_!/?,
%?"
 ?
inputs?????????`
? ")?&
?
0?????????`2
? ?
6__inference_WordEmbedding_evidence_layer_call_fn_22195R!/?,
%?"
 ?
inputs?????????`
? "??????????`2?
 __inference__wrapped_model_21632?!?BCX?U
N?K
I?F
!?
input_1?????????`
!?
input_2?????????`
? "-?*
(
dense?
dense??????????
H__inference_concatenate_1_layer_call_and_return_conditional_losses_22253?Z?W
P?M
K?H
"?
inputs/0?????????2
"?
inputs/1?????????
? "%?"
?
0?????????3
? ?
-__inference_concatenate_1_layer_call_fn_22246vZ?W
P?M
K?H
"?
inputs/0?????????2
"?
inputs/1?????????
? "??????????3?
H__inference_concatenate_2_layer_call_and_return_conditional_losses_22266?Z?W
P?M
K?H
"?
inputs/0?????????3
"?
inputs/1?????????3
? "%?"
?
0?????????f
? ?
-__inference_concatenate_2_layer_call_fn_22259vZ?W
P?M
K?H
"?
inputs/0?????????3
"?
inputs/1?????????3
? "??????????f?
F__inference_concatenate_layer_call_and_return_conditional_losses_22240?Z?W
P?M
K?H
"?
inputs/0?????????2
"?
inputs/1?????????
? "%?"
?
0?????????3
? ?
+__inference_concatenate_layer_call_fn_22233vZ?W
P?M
K?H
"?
inputs/0?????????2
"?
inputs/1?????????
? "??????????3?
@__inference_dense_layer_call_and_return_conditional_losses_22286\BC/?,
%?"
 ?
inputs?????????f
? "%?"
?
0?????????
? x
%__inference_dense_layer_call_fn_22275OBC/?,
%?"
 ?
inputs?????????f
? "???????????
A__inference_lambda_layer_call_and_return_conditional_losses_22221\3?0
)?&
?
inputs?????????

 
p 
? "%?"
?
0?????????
? ?
A__inference_lambda_layer_call_and_return_conditional_losses_22227\3?0
)?&
?
inputs?????????

 
p
? "%?"
?
0?????????
? y
&__inference_lambda_layer_call_fn_22210O3?0
)?&
?
inputs?????????

 
p 
? "??????????y
&__inference_lambda_layer_call_fn_22215O3?0
)?&
?
inputs?????????

 
p
? "???????????
@__inference_model_layer_call_and_return_conditional_losses_21959?!?BC`?]
V?S
I?F
!?
input_1?????????`
!?
input_2?????????`
p 

 
? "%?"
?
0?????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_22003?!?BC`?]
V?S
I?F
!?
input_1?????????`
!?
input_2?????????`
p

 
? "%?"
?
0?????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_22115?!?BCb?_
X?U
K?H
"?
inputs/0?????????`
"?
inputs/1?????????`
p 

 
? "%?"
?
0?????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_22171?!?BCb?_
X?U
K?H
"?
inputs/0?????????`
"?
inputs/1?????????`
p

 
? "%?"
?
0?????????
? ?
%__inference_model_layer_call_fn_21759?!?BC`?]
V?S
I?F
!?
input_1?????????`
!?
input_2?????????`
p 

 
? "???????????
%__inference_model_layer_call_fn_21915?!?BC`?]
V?S
I?F
!?
input_1?????????`
!?
input_2?????????`
p

 
? "???????????
%__inference_model_layer_call_fn_22043?!?BCb?_
X?U
K?H
"?
inputs/0?????????`
"?
inputs/1?????????`
p 

 
? "???????????
%__inference_model_layer_call_fn_22059?!?BCb?_
X?U
K?H
"?
inputs/0?????????`
"?
inputs/1?????????`
p

 
? "???????????
#__inference_signature_wrapper_22027?!?BCi?f
? 
_?\
,
input_1!?
input_1?????????`
,
input_2!?
input_2?????????`"-?*
(
dense?
dense?????????