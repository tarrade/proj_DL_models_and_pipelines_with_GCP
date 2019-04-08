
def del_all_flags(FLAGS, my_list):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        if keys in my_list:
            FLAGS.__delattr__(keys)