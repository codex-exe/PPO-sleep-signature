��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
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
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
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
dtypetype�
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
H
ShardedFilename
basename	
shard

num_shards
filename
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
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.11.02v2.11.0-rc2-15-g6290819256d8��
�
&Adam/v/critic_network_10/dense_65/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/v/critic_network_10/dense_65/bias
�
:Adam/v/critic_network_10/dense_65/bias/Read/ReadVariableOpReadVariableOp&Adam/v/critic_network_10/dense_65/bias*
_output_shapes
:*
dtype0
�
&Adam/m/critic_network_10/dense_65/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/m/critic_network_10/dense_65/bias
�
:Adam/m/critic_network_10/dense_65/bias/Read/ReadVariableOpReadVariableOp&Adam/m/critic_network_10/dense_65/bias*
_output_shapes
:*
dtype0
�
(Adam/v/critic_network_10/dense_65/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*9
shared_name*(Adam/v/critic_network_10/dense_65/kernel
�
<Adam/v/critic_network_10/dense_65/kernel/Read/ReadVariableOpReadVariableOp(Adam/v/critic_network_10/dense_65/kernel*
_output_shapes
:	�*
dtype0
�
(Adam/m/critic_network_10/dense_65/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*9
shared_name*(Adam/m/critic_network_10/dense_65/kernel
�
<Adam/m/critic_network_10/dense_65/kernel/Read/ReadVariableOpReadVariableOp(Adam/m/critic_network_10/dense_65/kernel*
_output_shapes
:	�*
dtype0
�
&Adam/v/critic_network_10/dense_64/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&Adam/v/critic_network_10/dense_64/bias
�
:Adam/v/critic_network_10/dense_64/bias/Read/ReadVariableOpReadVariableOp&Adam/v/critic_network_10/dense_64/bias*
_output_shapes	
:�*
dtype0
�
&Adam/m/critic_network_10/dense_64/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&Adam/m/critic_network_10/dense_64/bias
�
:Adam/m/critic_network_10/dense_64/bias/Read/ReadVariableOpReadVariableOp&Adam/m/critic_network_10/dense_64/bias*
_output_shapes	
:�*
dtype0
�
(Adam/v/critic_network_10/dense_64/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*9
shared_name*(Adam/v/critic_network_10/dense_64/kernel
�
<Adam/v/critic_network_10/dense_64/kernel/Read/ReadVariableOpReadVariableOp(Adam/v/critic_network_10/dense_64/kernel* 
_output_shapes
:
��*
dtype0
�
(Adam/m/critic_network_10/dense_64/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*9
shared_name*(Adam/m/critic_network_10/dense_64/kernel
�
<Adam/m/critic_network_10/dense_64/kernel/Read/ReadVariableOpReadVariableOp(Adam/m/critic_network_10/dense_64/kernel* 
_output_shapes
:
��*
dtype0
�
&Adam/v/critic_network_10/dense_63/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&Adam/v/critic_network_10/dense_63/bias
�
:Adam/v/critic_network_10/dense_63/bias/Read/ReadVariableOpReadVariableOp&Adam/v/critic_network_10/dense_63/bias*
_output_shapes	
:�*
dtype0
�
&Adam/m/critic_network_10/dense_63/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&Adam/m/critic_network_10/dense_63/bias
�
:Adam/m/critic_network_10/dense_63/bias/Read/ReadVariableOpReadVariableOp&Adam/m/critic_network_10/dense_63/bias*
_output_shapes	
:�*
dtype0
�
(Adam/v/critic_network_10/dense_63/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*9
shared_name*(Adam/v/critic_network_10/dense_63/kernel
�
<Adam/v/critic_network_10/dense_63/kernel/Read/ReadVariableOpReadVariableOp(Adam/v/critic_network_10/dense_63/kernel*
_output_shapes
:	�*
dtype0
�
(Adam/m/critic_network_10/dense_63/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*9
shared_name*(Adam/m/critic_network_10/dense_63/kernel
�
<Adam/m/critic_network_10/dense_63/kernel/Read/ReadVariableOpReadVariableOp(Adam/m/critic_network_10/dense_63/kernel*
_output_shapes
:	�*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
critic_network_10/dense_65/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!critic_network_10/dense_65/bias
�
3critic_network_10/dense_65/bias/Read/ReadVariableOpReadVariableOpcritic_network_10/dense_65/bias*
_output_shapes
:*
dtype0
�
!critic_network_10/dense_65/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*2
shared_name#!critic_network_10/dense_65/kernel
�
5critic_network_10/dense_65/kernel/Read/ReadVariableOpReadVariableOp!critic_network_10/dense_65/kernel*
_output_shapes
:	�*
dtype0
�
critic_network_10/dense_64/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*0
shared_name!critic_network_10/dense_64/bias
�
3critic_network_10/dense_64/bias/Read/ReadVariableOpReadVariableOpcritic_network_10/dense_64/bias*
_output_shapes	
:�*
dtype0
�
!critic_network_10/dense_64/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*2
shared_name#!critic_network_10/dense_64/kernel
�
5critic_network_10/dense_64/kernel/Read/ReadVariableOpReadVariableOp!critic_network_10/dense_64/kernel* 
_output_shapes
:
��*
dtype0
�
critic_network_10/dense_63/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*0
shared_name!critic_network_10/dense_63/bias
�
3critic_network_10/dense_63/bias/Read/ReadVariableOpReadVariableOpcritic_network_10/dense_63/bias*
_output_shapes	
:�*
dtype0
�
!critic_network_10/dense_63/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*2
shared_name#!critic_network_10/dense_63/kernel
�
5critic_network_10/dense_63/kernel/Read/ReadVariableOpReadVariableOp!critic_network_10/dense_63/kernel*
_output_shapes
:	�*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1!critic_network_10/dense_63/kernelcritic_network_10/dense_63/bias!critic_network_10/dense_64/kernelcritic_network_10/dense_64/bias!critic_network_10/dense_65/kernelcritic_network_10/dense_65/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *0
f+R)
'__inference_signature_wrapper_146159259

NoOpNoOp
�'
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�&
value�&B�& B�&
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
fc1
	fc2

q
	optimizer
loss

signatures*
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
* 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

kernel
bias*
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

kernel
bias*
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

kernel
bias*
�
/
_variables
0_iterations
1_learning_rate
2_index_dict
3
_momentums
4_velocities
5_update_step_xla*
* 

6serving_default* 
a[
VARIABLE_VALUE!critic_network_10/dense_63/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEcritic_network_10/dense_63/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!critic_network_10/dense_64/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEcritic_network_10/dense_64/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!critic_network_10/dense_65/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEcritic_network_10/dense_65/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
	1

2*
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
�
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

<trace_0* 

=trace_0* 

0
1*

0
1*
* 
�
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*

Ctrace_0* 

Dtrace_0* 

0
1*

0
1*
* 
�
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*

Jtrace_0* 

Ktrace_0* 
b
00
L1
M2
N3
O4
P5
Q6
R7
S8
T9
U10
V11
W12*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
L0
N1
P2
R3
T4
V5*
.
M0
O1
Q2
S3
U4
W5*
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
* 
sm
VARIABLE_VALUE(Adam/m/critic_network_10/dense_63/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE(Adam/v/critic_network_10/dense_63/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE&Adam/m/critic_network_10/dense_63/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE&Adam/v/critic_network_10/dense_63/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE(Adam/m/critic_network_10/dense_64/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE(Adam/v/critic_network_10/dense_64/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE&Adam/m/critic_network_10/dense_64/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE&Adam/v/critic_network_10/dense_64/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE(Adam/m/critic_network_10/dense_65/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/v/critic_network_10/dense_65/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/m/critic_network_10/dense_65/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/v/critic_network_10/dense_65/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename5critic_network_10/dense_63/kernel/Read/ReadVariableOp3critic_network_10/dense_63/bias/Read/ReadVariableOp5critic_network_10/dense_64/kernel/Read/ReadVariableOp3critic_network_10/dense_64/bias/Read/ReadVariableOp5critic_network_10/dense_65/kernel/Read/ReadVariableOp3critic_network_10/dense_65/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp<Adam/m/critic_network_10/dense_63/kernel/Read/ReadVariableOp<Adam/v/critic_network_10/dense_63/kernel/Read/ReadVariableOp:Adam/m/critic_network_10/dense_63/bias/Read/ReadVariableOp:Adam/v/critic_network_10/dense_63/bias/Read/ReadVariableOp<Adam/m/critic_network_10/dense_64/kernel/Read/ReadVariableOp<Adam/v/critic_network_10/dense_64/kernel/Read/ReadVariableOp:Adam/m/critic_network_10/dense_64/bias/Read/ReadVariableOp:Adam/v/critic_network_10/dense_64/bias/Read/ReadVariableOp<Adam/m/critic_network_10/dense_65/kernel/Read/ReadVariableOp<Adam/v/critic_network_10/dense_65/kernel/Read/ReadVariableOp:Adam/m/critic_network_10/dense_65/bias/Read/ReadVariableOp:Adam/v/critic_network_10/dense_65/bias/Read/ReadVariableOpConst*!
Tin
2	*
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
GPU 2J 8� *+
f&R$
"__inference__traced_save_146159442
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename!critic_network_10/dense_63/kernelcritic_network_10/dense_63/bias!critic_network_10/dense_64/kernelcritic_network_10/dense_64/bias!critic_network_10/dense_65/kernelcritic_network_10/dense_65/bias	iterationlearning_rate(Adam/m/critic_network_10/dense_63/kernel(Adam/v/critic_network_10/dense_63/kernel&Adam/m/critic_network_10/dense_63/bias&Adam/v/critic_network_10/dense_63/bias(Adam/m/critic_network_10/dense_64/kernel(Adam/v/critic_network_10/dense_64/kernel&Adam/m/critic_network_10/dense_64/bias&Adam/v/critic_network_10/dense_64/bias(Adam/m/critic_network_10/dense_65/kernel(Adam/v/critic_network_10/dense_65/kernel&Adam/m/critic_network_10/dense_65/bias&Adam/v/critic_network_10/dense_65/bias* 
Tin
2*
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
GPU 2J 8� *.
f)R'
%__inference__traced_restore_146159512��
�	
�
G__inference_dense_65_layer_call_and_return_conditional_losses_146159150

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
G__inference_dense_64_layer_call_and_return_conditional_losses_146159340

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
5__inference_critic_network_10_layer_call_fn_146159276	
state
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallstateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_critic_network_10_layer_call_and_return_conditional_losses_146159157o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:���������

_user_specified_namestate
�

�
G__inference_dense_64_layer_call_and_return_conditional_losses_146159134

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�\
�
%__inference__traced_restore_146159512
file_prefixE
2assignvariableop_critic_network_10_dense_63_kernel:	�A
2assignvariableop_1_critic_network_10_dense_63_bias:	�H
4assignvariableop_2_critic_network_10_dense_64_kernel:
��A
2assignvariableop_3_critic_network_10_dense_64_bias:	�G
4assignvariableop_4_critic_network_10_dense_65_kernel:	�@
2assignvariableop_5_critic_network_10_dense_65_bias:&
assignvariableop_6_iteration:	 *
 assignvariableop_7_learning_rate: N
;assignvariableop_8_adam_m_critic_network_10_dense_63_kernel:	�N
;assignvariableop_9_adam_v_critic_network_10_dense_63_kernel:	�I
:assignvariableop_10_adam_m_critic_network_10_dense_63_bias:	�I
:assignvariableop_11_adam_v_critic_network_10_dense_63_bias:	�P
<assignvariableop_12_adam_m_critic_network_10_dense_64_kernel:
��P
<assignvariableop_13_adam_v_critic_network_10_dense_64_kernel:
��I
:assignvariableop_14_adam_m_critic_network_10_dense_64_bias:	�I
:assignvariableop_15_adam_v_critic_network_10_dense_64_bias:	�O
<assignvariableop_16_adam_m_critic_network_10_dense_65_kernel:	�O
<assignvariableop_17_adam_v_critic_network_10_dense_65_kernel:	�H
:assignvariableop_18_adam_m_critic_network_10_dense_65_bias:H
:assignvariableop_19_adam_v_critic_network_10_dense_65_bias:
identity_21��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp2assignvariableop_critic_network_10_dense_63_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp2assignvariableop_1_critic_network_10_dense_63_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp4assignvariableop_2_critic_network_10_dense_64_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp2assignvariableop_3_critic_network_10_dense_64_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp4assignvariableop_4_critic_network_10_dense_65_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp2assignvariableop_5_critic_network_10_dense_65_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_iterationIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_learning_rateIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp;assignvariableop_8_adam_m_critic_network_10_dense_63_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp;assignvariableop_9_adam_v_critic_network_10_dense_63_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp:assignvariableop_10_adam_m_critic_network_10_dense_63_biasIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp:assignvariableop_11_adam_v_critic_network_10_dense_63_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp<assignvariableop_12_adam_m_critic_network_10_dense_64_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp<assignvariableop_13_adam_v_critic_network_10_dense_64_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp:assignvariableop_14_adam_m_critic_network_10_dense_64_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp:assignvariableop_15_adam_v_critic_network_10_dense_64_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp<assignvariableop_16_adam_m_critic_network_10_dense_65_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp<assignvariableop_17_adam_v_critic_network_10_dense_65_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp:assignvariableop_18_adam_m_critic_network_10_dense_65_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp:assignvariableop_19_adam_v_critic_network_10_dense_65_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_20Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_21IdentityIdentity_20:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_21Identity_21:output:0*=
_input_shapes,
*: : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
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
�

�
G__inference_dense_63_layer_call_and_return_conditional_losses_146159117

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
G__inference_dense_63_layer_call_and_return_conditional_losses_146159320

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
,__inference_dense_65_layer_call_fn_146159349

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_65_layer_call_and_return_conditional_losses_146159150o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�$
�
$__inference__wrapped_model_146159099
input_1L
9critic_network_10_dense_63_matmul_readvariableop_resource:	�I
:critic_network_10_dense_63_biasadd_readvariableop_resource:	�M
9critic_network_10_dense_64_matmul_readvariableop_resource:
��I
:critic_network_10_dense_64_biasadd_readvariableop_resource:	�L
9critic_network_10_dense_65_matmul_readvariableop_resource:	�H
:critic_network_10_dense_65_biasadd_readvariableop_resource:
identity��1critic_network_10/dense_63/BiasAdd/ReadVariableOp�0critic_network_10/dense_63/MatMul/ReadVariableOp�1critic_network_10/dense_64/BiasAdd/ReadVariableOp�0critic_network_10/dense_64/MatMul/ReadVariableOp�1critic_network_10/dense_65/BiasAdd/ReadVariableOp�0critic_network_10/dense_65/MatMul/ReadVariableOp�
0critic_network_10/dense_63/MatMul/ReadVariableOpReadVariableOp9critic_network_10_dense_63_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
!critic_network_10/dense_63/MatMulMatMulinput_18critic_network_10/dense_63/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
1critic_network_10/dense_63/BiasAdd/ReadVariableOpReadVariableOp:critic_network_10_dense_63_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
"critic_network_10/dense_63/BiasAddBiasAdd+critic_network_10/dense_63/MatMul:product:09critic_network_10/dense_63/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
critic_network_10/dense_63/ReluRelu+critic_network_10/dense_63/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
0critic_network_10/dense_64/MatMul/ReadVariableOpReadVariableOp9critic_network_10_dense_64_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
!critic_network_10/dense_64/MatMulMatMul-critic_network_10/dense_63/Relu:activations:08critic_network_10/dense_64/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
1critic_network_10/dense_64/BiasAdd/ReadVariableOpReadVariableOp:critic_network_10_dense_64_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
"critic_network_10/dense_64/BiasAddBiasAdd+critic_network_10/dense_64/MatMul:product:09critic_network_10/dense_64/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
critic_network_10/dense_64/ReluRelu+critic_network_10/dense_64/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
0critic_network_10/dense_65/MatMul/ReadVariableOpReadVariableOp9critic_network_10_dense_65_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
!critic_network_10/dense_65/MatMulMatMul-critic_network_10/dense_64/Relu:activations:08critic_network_10/dense_65/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
1critic_network_10/dense_65/BiasAdd/ReadVariableOpReadVariableOp:critic_network_10_dense_65_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
"critic_network_10/dense_65/BiasAddBiasAdd+critic_network_10/dense_65/MatMul:product:09critic_network_10/dense_65/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������z
IdentityIdentity+critic_network_10/dense_65/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp2^critic_network_10/dense_63/BiasAdd/ReadVariableOp1^critic_network_10/dense_63/MatMul/ReadVariableOp2^critic_network_10/dense_64/BiasAdd/ReadVariableOp1^critic_network_10/dense_64/MatMul/ReadVariableOp2^critic_network_10/dense_65/BiasAdd/ReadVariableOp1^critic_network_10/dense_65/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2f
1critic_network_10/dense_63/BiasAdd/ReadVariableOp1critic_network_10/dense_63/BiasAdd/ReadVariableOp2d
0critic_network_10/dense_63/MatMul/ReadVariableOp0critic_network_10/dense_63/MatMul/ReadVariableOp2f
1critic_network_10/dense_64/BiasAdd/ReadVariableOp1critic_network_10/dense_64/BiasAdd/ReadVariableOp2d
0critic_network_10/dense_64/MatMul/ReadVariableOp0critic_network_10/dense_64/MatMul/ReadVariableOp2f
1critic_network_10/dense_65/BiasAdd/ReadVariableOp1critic_network_10/dense_65/BiasAdd/ReadVariableOp2d
0critic_network_10/dense_65/MatMul/ReadVariableOp0critic_network_10/dense_65/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
P__inference_critic_network_10_layer_call_and_return_conditional_losses_146159300	
state:
'dense_63_matmul_readvariableop_resource:	�7
(dense_63_biasadd_readvariableop_resource:	�;
'dense_64_matmul_readvariableop_resource:
��7
(dense_64_biasadd_readvariableop_resource:	�:
'dense_65_matmul_readvariableop_resource:	�6
(dense_65_biasadd_readvariableop_resource:
identity��dense_63/BiasAdd/ReadVariableOp�dense_63/MatMul/ReadVariableOp�dense_64/BiasAdd/ReadVariableOp�dense_64/MatMul/ReadVariableOp�dense_65/BiasAdd/ReadVariableOp�dense_65/MatMul/ReadVariableOp�
dense_63/MatMul/ReadVariableOpReadVariableOp'dense_63_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0{
dense_63/MatMulMatMulstate&dense_63/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_63/BiasAdd/ReadVariableOpReadVariableOp(dense_63_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_63/BiasAddBiasAdddense_63/MatMul:product:0'dense_63/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_63/ReluReludense_63/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_64/MatMul/ReadVariableOpReadVariableOp'dense_64_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_64/MatMulMatMuldense_63/Relu:activations:0&dense_64/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_64/BiasAdd/ReadVariableOpReadVariableOp(dense_64_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_64/BiasAddBiasAdddense_64/MatMul:product:0'dense_64/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_64/ReluReludense_64/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_65/MatMul/ReadVariableOpReadVariableOp'dense_65_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_65/MatMulMatMuldense_64/Relu:activations:0&dense_65/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_65/BiasAdd/ReadVariableOpReadVariableOp(dense_65_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_65/BiasAddBiasAdddense_65/MatMul:product:0'dense_65/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_65/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_63/BiasAdd/ReadVariableOp^dense_63/MatMul/ReadVariableOp ^dense_64/BiasAdd/ReadVariableOp^dense_64/MatMul/ReadVariableOp ^dense_65/BiasAdd/ReadVariableOp^dense_65/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2B
dense_63/BiasAdd/ReadVariableOpdense_63/BiasAdd/ReadVariableOp2@
dense_63/MatMul/ReadVariableOpdense_63/MatMul/ReadVariableOp2B
dense_64/BiasAdd/ReadVariableOpdense_64/BiasAdd/ReadVariableOp2@
dense_64/MatMul/ReadVariableOpdense_64/MatMul/ReadVariableOp2B
dense_65/BiasAdd/ReadVariableOpdense_65/BiasAdd/ReadVariableOp2@
dense_65/MatMul/ReadVariableOpdense_65/MatMul/ReadVariableOp:N J
'
_output_shapes
:���������

_user_specified_namestate
�
�
P__inference_critic_network_10_layer_call_and_return_conditional_losses_146159157	
state%
dense_63_146159118:	�!
dense_63_146159120:	�&
dense_64_146159135:
��!
dense_64_146159137:	�%
dense_65_146159151:	� 
dense_65_146159153:
identity�� dense_63/StatefulPartitionedCall� dense_64/StatefulPartitionedCall� dense_65/StatefulPartitionedCall�
 dense_63/StatefulPartitionedCallStatefulPartitionedCallstatedense_63_146159118dense_63_146159120*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_63_layer_call_and_return_conditional_losses_146159117�
 dense_64/StatefulPartitionedCallStatefulPartitionedCall)dense_63/StatefulPartitionedCall:output:0dense_64_146159135dense_64_146159137*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_64_layer_call_and_return_conditional_losses_146159134�
 dense_65/StatefulPartitionedCallStatefulPartitionedCall)dense_64/StatefulPartitionedCall:output:0dense_65_146159151dense_65_146159153*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_65_layer_call_and_return_conditional_losses_146159150x
IdentityIdentity)dense_65/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_63/StatefulPartitionedCall!^dense_64/StatefulPartitionedCall!^dense_65/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2D
 dense_63/StatefulPartitionedCall dense_63/StatefulPartitionedCall2D
 dense_64/StatefulPartitionedCall dense_64/StatefulPartitionedCall2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall:N J
'
_output_shapes
:���������

_user_specified_namestate
�
�
P__inference_critic_network_10_layer_call_and_return_conditional_losses_146159238
input_1%
dense_63_146159222:	�!
dense_63_146159224:	�&
dense_64_146159227:
��!
dense_64_146159229:	�%
dense_65_146159232:	� 
dense_65_146159234:
identity�� dense_63/StatefulPartitionedCall� dense_64/StatefulPartitionedCall� dense_65/StatefulPartitionedCall�
 dense_63/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_63_146159222dense_63_146159224*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_63_layer_call_and_return_conditional_losses_146159117�
 dense_64/StatefulPartitionedCallStatefulPartitionedCall)dense_63/StatefulPartitionedCall:output:0dense_64_146159227dense_64_146159229*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_64_layer_call_and_return_conditional_losses_146159134�
 dense_65/StatefulPartitionedCallStatefulPartitionedCall)dense_64/StatefulPartitionedCall:output:0dense_65_146159232dense_65_146159234*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_65_layer_call_and_return_conditional_losses_146159150x
IdentityIdentity)dense_65/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_63/StatefulPartitionedCall!^dense_64/StatefulPartitionedCall!^dense_65/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2D
 dense_63/StatefulPartitionedCall dense_63/StatefulPartitionedCall2D
 dense_64/StatefulPartitionedCall dense_64/StatefulPartitionedCall2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
,__inference_dense_63_layer_call_fn_146159309

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_63_layer_call_and_return_conditional_losses_146159117p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
'__inference_signature_wrapper_146159259
input_1
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference__wrapped_model_146159099o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
,__inference_dense_64_layer_call_fn_146159329

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_64_layer_call_and_return_conditional_losses_146159134p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
5__inference_critic_network_10_layer_call_fn_146159172
input_1
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_critic_network_10_layer_call_and_return_conditional_losses_146159157o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�5
�
"__inference__traced_save_146159442
file_prefix@
<savev2_critic_network_10_dense_63_kernel_read_readvariableop>
:savev2_critic_network_10_dense_63_bias_read_readvariableop@
<savev2_critic_network_10_dense_64_kernel_read_readvariableop>
:savev2_critic_network_10_dense_64_bias_read_readvariableop@
<savev2_critic_network_10_dense_65_kernel_read_readvariableop>
:savev2_critic_network_10_dense_65_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableopG
Csavev2_adam_m_critic_network_10_dense_63_kernel_read_readvariableopG
Csavev2_adam_v_critic_network_10_dense_63_kernel_read_readvariableopE
Asavev2_adam_m_critic_network_10_dense_63_bias_read_readvariableopE
Asavev2_adam_v_critic_network_10_dense_63_bias_read_readvariableopG
Csavev2_adam_m_critic_network_10_dense_64_kernel_read_readvariableopG
Csavev2_adam_v_critic_network_10_dense_64_kernel_read_readvariableopE
Asavev2_adam_m_critic_network_10_dense_64_bias_read_readvariableopE
Asavev2_adam_v_critic_network_10_dense_64_bias_read_readvariableopG
Csavev2_adam_m_critic_network_10_dense_65_kernel_read_readvariableopG
Csavev2_adam_v_critic_network_10_dense_65_kernel_read_readvariableopE
Asavev2_adam_m_critic_network_10_dense_65_bias_read_readvariableopE
Asavev2_adam_v_critic_network_10_dense_65_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
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
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0<savev2_critic_network_10_dense_63_kernel_read_readvariableop:savev2_critic_network_10_dense_63_bias_read_readvariableop<savev2_critic_network_10_dense_64_kernel_read_readvariableop:savev2_critic_network_10_dense_64_bias_read_readvariableop<savev2_critic_network_10_dense_65_kernel_read_readvariableop:savev2_critic_network_10_dense_65_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableopCsavev2_adam_m_critic_network_10_dense_63_kernel_read_readvariableopCsavev2_adam_v_critic_network_10_dense_63_kernel_read_readvariableopAsavev2_adam_m_critic_network_10_dense_63_bias_read_readvariableopAsavev2_adam_v_critic_network_10_dense_63_bias_read_readvariableopCsavev2_adam_m_critic_network_10_dense_64_kernel_read_readvariableopCsavev2_adam_v_critic_network_10_dense_64_kernel_read_readvariableopAsavev2_adam_m_critic_network_10_dense_64_bias_read_readvariableopAsavev2_adam_v_critic_network_10_dense_64_bias_read_readvariableopCsavev2_adam_m_critic_network_10_dense_65_kernel_read_readvariableopCsavev2_adam_v_critic_network_10_dense_65_kernel_read_readvariableopAsavev2_adam_m_critic_network_10_dense_65_bias_read_readvariableopAsavev2_adam_v_critic_network_10_dense_65_bias_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *#
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :	�:�:
��:�:	�:: : :	�:	�:�:�:
��:
��:�:�:	�:	�::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :%	!

_output_shapes
:	�:%
!

_output_shapes
:	�:!

_output_shapes	
:�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:!

_output_shapes	
:�:%!

_output_shapes
:	�:%!

_output_shapes
:	�: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
�	
�
G__inference_dense_65_layer_call_and_return_conditional_losses_146159359

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:�_
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
fc1
	fc2

q
	optimizer
loss

signatures"
_tf_keras_model
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
trace_0
trace_12�
5__inference_critic_network_10_layer_call_fn_146159172
5__inference_critic_network_10_layer_call_fn_146159276�
���
FullArgSpec
args�
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0ztrace_1
�
trace_0
trace_12�
P__inference_critic_network_10_layer_call_and_return_conditional_losses_146159300
P__inference_critic_network_10_layer_call_and_return_conditional_losses_146159238�
���
FullArgSpec
args�
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0ztrace_1
�B�
$__inference__wrapped_model_146159099input_1"�
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
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
/
_variables
0_iterations
1_learning_rate
2_index_dict
3
_momentums
4_velocities
5_update_step_xla"
experimentalOptimizer
 "
trackable_dict_wrapper
,
6serving_default"
signature_map
4:2	�2!critic_network_10/dense_63/kernel
.:,�2critic_network_10/dense_63/bias
5:3
��2!critic_network_10/dense_64/kernel
.:,�2critic_network_10/dense_64/bias
4:2	�2!critic_network_10/dense_65/kernel
-:+2critic_network_10/dense_65/bias
 "
trackable_list_wrapper
5
0
	1

2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
5__inference_critic_network_10_layer_call_fn_146159172input_1"�
���
FullArgSpec
args�
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
5__inference_critic_network_10_layer_call_fn_146159276state"�
���
FullArgSpec
args�
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_critic_network_10_layer_call_and_return_conditional_losses_146159300state"�
���
FullArgSpec
args�
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_critic_network_10_layer_call_and_return_conditional_losses_146159238input_1"�
���
FullArgSpec
args�
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
�
<trace_02�
,__inference_dense_63_layer_call_fn_146159309�
���
FullArgSpec
args�
jself
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
annotations� *
 z<trace_0
�
=trace_02�
G__inference_dense_63_layer_call_and_return_conditional_losses_146159320�
���
FullArgSpec
args�
jself
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
annotations� *
 z=trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
�
Ctrace_02�
,__inference_dense_64_layer_call_fn_146159329�
���
FullArgSpec
args�
jself
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
annotations� *
 zCtrace_0
�
Dtrace_02�
G__inference_dense_64_layer_call_and_return_conditional_losses_146159340�
���
FullArgSpec
args�
jself
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
annotations� *
 zDtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
�
Jtrace_02�
,__inference_dense_65_layer_call_fn_146159349�
���
FullArgSpec
args�
jself
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
annotations� *
 zJtrace_0
�
Ktrace_02�
G__inference_dense_65_layer_call_and_return_conditional_losses_146159359�
���
FullArgSpec
args�
jself
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
annotations� *
 zKtrace_0
~
00
L1
M2
N3
O4
P5
Q6
R7
S8
T9
U10
V11
W12"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
J
L0
N1
P2
R3
T4
V5"
trackable_list_wrapper
J
M0
O1
Q2
S3
U4
W5"
trackable_list_wrapper
�2��
���
FullArgSpec2
args*�'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
'__inference_signature_wrapper_146159259input_1"�
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
,__inference_dense_63_layer_call_fn_146159309inputs"�
���
FullArgSpec
args�
jself
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
annotations� *
 
�B�
G__inference_dense_63_layer_call_and_return_conditional_losses_146159320inputs"�
���
FullArgSpec
args�
jself
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
annotations� *
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
,__inference_dense_64_layer_call_fn_146159329inputs"�
���
FullArgSpec
args�
jself
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
annotations� *
 
�B�
G__inference_dense_64_layer_call_and_return_conditional_losses_146159340inputs"�
���
FullArgSpec
args�
jself
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
annotations� *
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
,__inference_dense_65_layer_call_fn_146159349inputs"�
���
FullArgSpec
args�
jself
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
annotations� *
 
�B�
G__inference_dense_65_layer_call_and_return_conditional_losses_146159359inputs"�
���
FullArgSpec
args�
jself
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
annotations� *
 
9:7	�2(Adam/m/critic_network_10/dense_63/kernel
9:7	�2(Adam/v/critic_network_10/dense_63/kernel
3:1�2&Adam/m/critic_network_10/dense_63/bias
3:1�2&Adam/v/critic_network_10/dense_63/bias
::8
��2(Adam/m/critic_network_10/dense_64/kernel
::8
��2(Adam/v/critic_network_10/dense_64/kernel
3:1�2&Adam/m/critic_network_10/dense_64/bias
3:1�2&Adam/v/critic_network_10/dense_64/bias
9:7	�2(Adam/m/critic_network_10/dense_65/kernel
9:7	�2(Adam/v/critic_network_10/dense_65/kernel
2:02&Adam/m/critic_network_10/dense_65/bias
2:02&Adam/v/critic_network_10/dense_65/bias�
$__inference__wrapped_model_146159099o0�-
&�#
!�
input_1���������
� "3�0
.
output_1"�
output_1����������
P__inference_critic_network_10_layer_call_and_return_conditional_losses_146159238h0�-
&�#
!�
input_1���������
� ",�)
"�
tensor_0���������
� �
P__inference_critic_network_10_layer_call_and_return_conditional_losses_146159300f.�+
$�!
�
state���������
� ",�)
"�
tensor_0���������
� �
5__inference_critic_network_10_layer_call_fn_146159172]0�-
&�#
!�
input_1���������
� "!�
unknown����������
5__inference_critic_network_10_layer_call_fn_146159276[.�+
$�!
�
state���������
� "!�
unknown����������
G__inference_dense_63_layer_call_and_return_conditional_losses_146159320d/�,
%�"
 �
inputs���������
� "-�*
#� 
tensor_0����������
� �
,__inference_dense_63_layer_call_fn_146159309Y/�,
%�"
 �
inputs���������
� ""�
unknown�����������
G__inference_dense_64_layer_call_and_return_conditional_losses_146159340e0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
,__inference_dense_64_layer_call_fn_146159329Z0�-
&�#
!�
inputs����������
� ""�
unknown�����������
G__inference_dense_65_layer_call_and_return_conditional_losses_146159359d0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������
� �
,__inference_dense_65_layer_call_fn_146159349Y0�-
&�#
!�
inputs����������
� "!�
unknown����������
'__inference_signature_wrapper_146159259z;�8
� 
1�.
,
input_1!�
input_1���������"3�0
.
output_1"�
output_1���������