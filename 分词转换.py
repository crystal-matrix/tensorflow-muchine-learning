import  jieba
fin = open('medical.txt', 'r')
#s=f1.read()
fou = open('medicalresout.txt', 'w')
line=fin.readline()
while line:
  #newline=jieba.cut(line,cut_all=True, HMM=False)
  str_out=''.join(line).replace('，',' ').replace('。',' ').replace('？',' ').replace('！',' ')\
             .replace('：','').replace('“.','').replace('”','').replace('‘','').replace('’','').replace('-','')\
             .replace('（','').replace('）','').replace('《','').replace('》','').replace('；','').replace('.','')\
             .replace('、','').replace('。。。','').replace('......','').replace('.','').replace(',','').replace('!','')\
             .replace(':','').replace('"','').replace('""','').replace('-','').replace('!!!','')\
             .replace('(','').replace(')','').replace('<','').replace('>','').replace(';','').replace('...','')\
            .replace('<<','').replace('>>','')
#print(str_out)
print>> fou,str_out,
#line=fin.readline()
fin.close()
fou.close()