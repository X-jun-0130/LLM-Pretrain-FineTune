import random

def joint(txt_list):
    new_txtlist = []
    point = []
    n = 0
    while len(point) < len(txt_list) and n < len(txt_list):
        if n not in point:
            joint_str = txt_list[n]
            point.append(n)

            for j in range(n+1, len(txt_list)):
                if j not in point:
                    if len(joint_str + '###' + txt_list[j]) > 1080:
                        pass
                    else:
                        joint_str +=  '###' + txt_list[j]
                        point.append(j)
            new_txtlist.append(joint_str)
        else:
            pass

        n += 1
    return new_txtlist



def chunk_1024(file_list):
    data_list = []
    n = 0
    _list = []
    for f in file_list:
        _list += f[0][:f[1]]
    random.shuffle(_list)
    n += len(_list)

    data_list.extend([k for k in _list if len(k) >= 1000])
    '''
    时间太长了，将列表分20批进行组合
    '''
    list_1080 = [k for k in _list if len(k) < 1000]
    length = len(list_1080)
    slice = 20
    m = int(length/slice)+1
    for i in range(slice):
        slice_list = list_1080[i*m:(i+1)*m]
        joint_1080 = joint(slice_list)
        data_list.extend(joint_1080)
    
    return data_list, n
