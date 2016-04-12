# -*- coding: utf-8 -*-
'''
Created on 2016年4月11日

@author: xhj
'''
import os

import nltk
from nltk.corpus import stopwords
import codecs
from pprint import pprint
import math
import random

# 所有的文件,key为文件的类别，value为一个文件对象
all_files = {}
# 使用 wordnet 进行词干还原
wnl = nltk.WordNetLemmatizer()
# 使用 NLTK 中的停用词表，去除停用词
english_stopwords = stopwords.words( 'english' )

# 一个文件类，
# From,Subject,Summary,Keywords,Expires,Distribution,Organization,Supersedes,Lines,content,class_lable
class OneFile():
     
    def __init__( self, content , class_lable, From = "", Subject = "", Summary = "", Keywords = "", Expires = "", \
              Distribution = "", Organization = "", Supersedes = "", Lines = "" ):
        self.From = From
        self.Subject = Subject
        self.Summary = Summary
        self.Keywords = Keywords
        self.Expires = Expires
        self.Distribution = Distribution
        self.Organization = Organization
        self.Supersedes = Supersedes
        self.Lines = Lines
        self.content = content
        self.class_lable = class_lable        
    
    def print_to_console( self ):
        print 'From', self.From 
        print 'Subject', self.Subject 
        print 'Summary', self.Summary  
        print 'Keywords', self.Keywords  
        print 'Expires', self.Expires 
        print 'Distribution', self.Distribution  
        print 'Organization', self.Organization  
        print 'Supersedes', self.Supersedes  
        print 'Lines', self.Lines 
        print 'content', self.content 
        print 'class_lable', self.class_lable         
     
     
# 读取所有的数据    
def read_all_datas():
    global all_files        
    for one_folder in os.listdir( "./origin_datas" ):
        all_files[one_folder] = []
        for one_file in os.listdir( "./origin_datas/" + one_folder ):
            # 开始处理一篇文档
            konghang_pass = False
            From = Subject = Summary = Keywords = Expires = Distribution = Organization = Supersedes = Lines = content = ""
            class_lable = one_folder
            for line in codecs.open( "./origin_datas/" + one_folder + "/" + one_file, "r", encoding = 'utf-8', errors = 'ignore' ).readlines():
                line = line[:-1]
                if not konghang_pass:
                    tag = line[:line.find( ': ' )]
                    tag_text = segmenter_and_process( line[line.find( ': ' ) + 1:] )
                    if tag == "From":
                        From = tag_text
                    if tag == "Subject":
                        Subject = tag_text
                    if tag == "Summary":
                        Summary = tag_text
                    if tag == "Keywords":
                        Keywords = tag_text
                    if tag == "Expires":
                        Expires = tag_text
                    if tag == "Distribution":
                        Distribution = tag_text
                    if tag == "Organization":
                        Organization = tag_text
                    if tag == "Supersedes":
                        Supersedes = tag_text
                    if tag == "Lines":
                        Lines = tag_text
                    if len( line ) == 0:
                        konghang_pass = True
                else:
                    content = content + " " + line
            content = segmenter_and_process( content )
            all_files[one_folder].append( OneFile( content , class_lable , From, Subject, Summary, Keywords, Expires , Distribution , Organization , Supersedes , Lines ) )


# 对一段文本进行分词，和词干还原，去除停用词，去除非单词
def segmenter_and_process( content ):
    global wnl
    segmenter_text = []
    sentences = [nltk.word_tokenize( sent ) for sent in nltk.sent_tokenize( content )]
    for sent in sentences:
        for one_word in sent:
            is_word = not_stop_words = True
            if one_word in english_stopwords:
                not_stop_words = False
            if not_stop_words:
                for one_char in one_word:
                    if not ( one_char >= 'a' and one_char <= 'z' or one_char >= 'A' and one_char <= 'Z' ):
                        is_word = False
                        break
            if is_word and not_stop_words:
                segmenter_text.append( wnl.lemmatize( one_word ) )
    return segmenter_text


# 处理一个 list 如： [1,2,3,4,1,2,3,5]
# 返回：{1:2,2:2,3:3,4:1,5:1}
def pro_list( old_list ):
    new_dict = {}
    for one in old_list:
        new_dict[one] = new_dict.get( one, 0 ) + 1
    for one_key in new_dict:
        new_dict[one_key] = float( new_dict[one_key] ) / len( old_list ) 
    return new_dict


# 对每个类别下的词语，计算 TF IDF 
def pro_all_files():
    global all_files        
    all_files_word_dict = {}
    for one_class in all_files:
        all_files_word_dict[one_class] = []
        for one_obj in all_files[one_class]:
            all_files_word_dict[one_class].extend( one_obj.Subject )
            all_files_word_dict[one_class].extend( one_obj.Summary )
            all_files_word_dict[one_class].extend( one_obj.Keywords )
            all_files_word_dict[one_class].extend( one_obj.Organization )
            all_files_word_dict[one_class].extend( one_obj.content )
        all_files_word_dict[one_class] = pro_list( all_files_word_dict[one_class] )
        
    idf_all_words_map = {}
    for one_class in all_files_word_dict:
        for one_word in all_files_word_dict[one_class]:
            idf_all_words_map[one_word] = idf_all_words_map.get( one_word, 0 ) + 1
    for one_word in idf_all_words_map:
        idf_all_words_map[one_word] = math.log( 20.0 / float( idf_all_words_map[one_word] ) * 2 )
     
    for one_class in all_files_word_dict:
        for one_word in all_files_word_dict[one_class]:
            all_files_word_dict[one_class][one_word] = all_files_word_dict[one_class][one_word] * idf_all_words_map[one_word]
    
#     result_file = open("result_file.txt",'w')
#     pprint(all_files_word_dict,result_file)
    return all_files_word_dict


def predict_onefile( one_obj , all_files_word_dict ):
    high_words = []
    normal_words = []
    
    high_words.extend( one_obj.Subject )
    high_words.extend( one_obj.Summary )
    high_words.extend( one_obj.Keywords )
    high_words.extend( one_obj.Organization )
    normal_words.extend( one_obj.content )
    
    posibility_classlable = {}
    for one_class in all_files_word_dict:
        posibility_classlable[one_class] = 0.0
        for one_word in high_words:
            posibility_classlable[one_class] = posibility_classlable[one_class] + 15 * all_files_word_dict[one_class].get( one_word, -0.1 )
        for one_word in normal_words:
            posibility_classlable[one_class] = posibility_classlable[one_class] + all_files_word_dict[one_class].get( one_word, -0.1 )
        posibility_classlable[one_class] = posibility_classlable[one_class] / ( len( high_words ) + len( normal_words ) )
    
    predict_class = ""
    max_pos = 0.0
    for one_class in posibility_classlable:
        if max_pos < posibility_classlable[one_class]:
            max_pos = posibility_classlable[one_class]
            predict_class = one_class
    return predict_class


def test_precision( all_files_word_dict ):
    test_files = {}
    global all_files        
    for one_class in all_files:
        random.shuffle( all_files[one_class] )
        test_files[one_class] = all_files[one_class][:int( len( all_files[one_class] ) * 0.3 )]
    
    precision_class = {}    
    for one_class in test_files:
        for one_file in test_files[one_class]:
            pre_lable = predict_onefile( one_file, all_files_word_dict )
            precision_class[one_class] = precision_class.get( one_class, 0.0 ) + \
             1.0 if one_class == pre_lable else 0.0
        precision_class[one_class] = float( precision_class[one_class] ) / len( test_files[one_class] )
    
    pprint( precision_class )
    pass



if __name__ == '__main__':
    read_all_datas()
    all_files_word_dict = pro_all_files()
    test_precision( all_files_word_dict )
    
    pass
