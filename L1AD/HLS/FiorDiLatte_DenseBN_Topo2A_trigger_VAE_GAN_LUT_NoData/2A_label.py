import os

projectname = "Topo2A_AD_proj" 
# File paths
myproj_h = 'firmware/myproject.h'
myproj_cpp = 'firmware/myproject.cpp'
defines = 'firmware/defines.h'
myproj_tb = './myproject_test.cpp'

myproj_h = myproj_h.replace("myproject", projectname)
myproj_cpp = myproj_cpp.replace("myproject", projectname)
myproj_tb = myproj_tb.replace("myproject", projectname)

# Configuration
feats = ['pt', 'eta', 'phi']
nfeat = len(feats)
njets = 6
negms = 0
netaus = 4 
njtaus = 0 
nmus = 4 
nmet = 1
ntotal = njets + negms + netaus + njtaus + nmus + nmet
assert njets > 0

# Backup and restore
start_from_scratch = True 
if start_from_scratch:
    os.system('cp -Tr firmware_bak firmware')
    os.system(f'cp {myproj_tb}.bak {myproj_tb}')
os.system('cp -Tr firmware firmware_bak')
os.system(f'cp {myproj_tb} {myproj_tb}.bak')

# Process myproject.h to get nameinput
with open(myproj_h, 'r') as file:
    lines = file.readlines()
for line in lines:
    if line.startswith("    input_t input"):
        nameinput = line.replace("    input_t inputs[", "").replace("],", "").rstrip()
        break
print(f"nameinput: {nameinput}")

# Process defines.h
with open(defines, 'r') as file:
    lines = file.readlines()
for line in lines:
    if line.startswith(f"#define {nameinput}"):
        ninputs = int(line.replace(f"#define {nameinput} ", "").rstrip())
    if line.rstrip().endswith("input_t;"):
        bitwidth = [int(x) for x in line.replace("typedef ap_fixed<", "").replace("> input_t;", "").rstrip().split(",")]
print(f"ninputs: {ninputs}")
print(f"bitwidth: {bitwidth}")
assert ninputs == (ntotal * nfeat - nmet)  # Subtract nmet because met doesn't have eta

# Generate object types and indices
objtype = []
objind = []
for j in range(ntotal):
    if j < njets:
        objtype.append("jet")
        objind.append(str(j))
    elif j < njets + negms:
        objtype.append("egm")
        objind.append(str(j - njets))
    elif j < njets + negms + netaus:
        objtype.append("etau")
        objind.append(str(j - njets - negms))
    elif j < njets + negms + netaus + njtaus:
        objtype.append("jtau")
        objind.append(str(j - njets - negms - netaus))
    elif j < njets + negms + netaus + njtaus + nmus:
        objtype.append("mu")
        objind.append(str(j - njets - negms - netaus - njtaus))
    else:
        objtype.append("met")
        objind.append(str(j - njets - negms - netaus - njtaus - nmus))
print(f"objtype: {objtype}")
print(f"objind: {objind}")

# Helper function to generate input variables
def generate_input_vars():
    vars = []
    for j in range(ntotal):
        for k in range(nfeat):
            if objtype[j] == "met" and feats[k] == "eta":
                continue
            vars.append(f"in_{objtype[j]}_{feats[k]}_{objind[j]}")
    return vars

# Modify myproject.h
with open(myproj_h, 'r') as file:
    lines = file.readlines()

for i, line in enumerate(lines):
    if line.startswith("    input_t inputs"):
        lines[i] = "    input_t in_jet_pt_0,\n"
        for var in generate_input_vars()[1:]:  # Skip the first one as it's already added
            lines.insert(i + 1, f"    input_t {var},\n")
            i += 1
        break

with open(myproj_h, 'w') as file:
    file.writelines(lines)

# Modify myproject.cpp
with open(myproj_cpp, 'r') as file:
    lines = file.readlines()

modified_line = ""
for i, line in enumerate(lines):
    if line.startswith("    input_t inputs"):
        modified_line = line.replace(",", ";")
        lines[i] = "    input_t in_jet_pt_0,\n"
    if "port=inputs" in line:
        lines[i] = line.replace("inputs", ",".join(generate_input_vars()))

i = 0
while i < len(lines):
    if "    input_t in_jet_pt_0," in lines[i]:
        for var in generate_input_vars()[1:]:  # Skip the first one as it's already added
            lines.insert(i + 1, f"    input_t {var},\n")
            i += 1
    if ") {" in lines[i]:
        lines.insert(i + 1, modified_line)
        i += 1
    if "pragma HLS PIPELINE" in lines[i]:
        input_index = 0
        for j in range(ntotal):
            for k in range(nfeat):
                if objtype[j] == "met" and feats[k] == "eta":
                    continue
                lines.insert(i + 1, f"    inputs[{input_index}] = in_{objtype[j]}_{feats[k]}_{objind[j]};\n")
                i += 1
                input_index += 1
    i += 1

with open(myproj_cpp, 'w') as file:
    file.writelines(lines)

# Modify myproject_test.cpp
with open(myproj_tb, 'r') as file:
    lines = file.readlines()

findline = "            myproject(inputs"
findline = findline.replace("myproject", projectname)
for i, line in enumerate(lines):
    if line.startswith(findline):
        lines[i] = line.replace("inputs", ",".join(generate_input_vars()))

i = 0
input_index = 0
while i < len(lines):
    if "    nnet::copy_data" in lines[i] or "    nnet::fill_zero" in lines[i]:
        for j in range(ntotal):
            for k in range(nfeat):
                if objtype[j] == "met" and feats[k] == "eta":
                    continue
                lines.insert(i + 1, f"      input_t in_{objtype[j]}_{feats[k]}_{objind[j]} = inputs[{input_index}];\n")
                i += 1
                input_index += 1
    i += 1

with open(myproj_tb, 'w') as file:
    file.writelines(lines)

print("Script execution completed.")
