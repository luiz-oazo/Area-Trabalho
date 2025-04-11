#!/usr/bin/env python
# coding: utf-8

## LIST   []


fruits = ['orange', 'apple', 'pear', 'banana', 'kiwi', 'apple', 'banana']
type(fruits)


# In[102]:


len(fruits)


# In[103]:


fruits[0]


# In[104]:


fruits[-7]


# In[105]:


fruits[:]


# In[106]:


fruits[2:5]


# In[4]:


a = [4,65,7,3,5,6,8,3,2,1,6]
a


# In[5]:


b = [1, "p", 5, 8.999, 74]
b


# In[9]:


type(b[3])


# In[7]:


type(b)


# In[21]:


fruits = ['orange', 'apple', 'pear', 'banana', 'kiwi', 'apple', 'banana']
type(fruits)


# In[22]:


fruits.append("pineapple")


# In[23]:


fruits


# In[24]:


fruits.count('banana')


# In[25]:


fruits.remove("apple")
fruits


# In[32]:


fruits.index("banana")


# In[29]:


fruits.pop(2)
fruits


# In[39]:


fruits.sort()
fruits


# In[42]:


fruits.reverse()
fruits


# In[44]:


fruits[2:6]


# In[45]:


numeros = [0, 1, 2, 3, 4, 5]


# In[47]:


x = []
for i in numeros:
    x.append(i**2)
print(x)


# In[49]:


y = [x**2 for x in numeros]
print(y)


# In[ ]:


## Tupla ()


# In[64]:


b = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
b


# In[54]:


type(b[0])


# In[61]:


b = [[1, 0, 0], (0, 1, 0), (0, 0, 1)]
b


# In[62]:


b[1][2]


# In[55]:


a = (1, 2, 3)


# In[56]:


b = [1, 2, 3]


# In[57]:


a


# In[60]:


a.remove(1)


# In[58]:


b


# In[59]:


b.remove(1)
b


# In[65]:


tarifa = [("onibus", "SP", 10), ("onibus", "RJ", 9), 
          ("onibus", "ZZ", 5), ("taxi", "RJ", 15),
          ("taxi", "AL", 50),
          ("taxi", "SP", 20), ("taxi", "ZZ", 30)]


# In[66]:


tarifa


# In[70]:


tipo = input("tipo de transporte: ")
estado = input("Estado: ")

q = 0
for i in tarifa:
    q+=1
    if i[0] == tipo:
        if i[1] == estado:
            print("tarifa", estado, i[2])
            break
        else:
            if i[1] == "ZZ":
                print("tarifa", estado, i[2])
                break
print("registros lidos", q)


# In[107]:


## DICT


# In[99]:


tarifa_dict = {"onibus": {'RJ': 9,
                          'SP': 10,
                          'ZZ': 5},
               "taxi": {'RJ': 15,
                        'SP': 20,
                        'AL': 50,
                        'ZZ': 30}}
tarifa_dict


# In[97]:


del tarifa_dict["onibus"]


# In[98]:


tarifa_dict


# In[92]:


tarifa_dict.items()


# In[95]:


tarifa_dict.values()


# In[94]:


tarifa_dict['onibus'].values()


# In[74]:


tarifa_dict.keys()


# In[75]:


tarifa_dict["onibus"].keys()


# In[77]:


tarifa_dict["onibus"]['SP']


# In[78]:


tarifa_dict.get("onibus")


# In[83]:


x = tarifa_dict["onibus"].get("DD")
if not x:
    print("ok")


# In[84]:


tipo = input("tipo de transporte: ")
estado = input("Estado: ")

if tarifa_dict.get(tipo):
    if tarifa_dict[tipo].get(estado):
        print(estado, tarifa_dict[tipo].get(estado))
    else:
        print(estado, tarifa_dict[tipo].get("ZZ"))


# In[87]:


estado = input("Estado: ")

for tipo in tarifa_dict.keys():
    if tarifa_dict[tipo].get(estado):
        print(estado, tipo)


# In[109]:


## SET   {}


# In[131]:


frutas = {"maçã", "banana", "cereja"}
type(frutas)


# In[132]:


frutas


# In[133]:


frutas.add("laranja")
frutas


# In[134]:


frutas.add("abacate")
frutas


# In[126]:


frutas.remove("banana")
frutas


# In[127]:


frutas.discard("banana")
frutas


# In[128]:


frutas.discard("laranja")
frutas


# In[135]:


fruta_removida = frutas.pop()
fruta_removida


# In[136]:


frutas


# In[138]:


frutas.clear()
frutas


# In[139]:


conjunto1 = {1, 2, 3}
conjunto2 = {3, 4, 5}
uniao = conjunto1.union(conjunto2)
uniao


# In[140]:


intersecao = conjunto1.intersection(conjunto2)
intersecao


# In[142]:


diferenca = conjunto1.difference(conjunto2)
diferenca


# In[143]:


diferenca = conjunto2.difference(conjunto1)
diferenca


# In[144]:


subconjunto = {1, 2}
resultado = subconjunto.issubset(conjunto1)
resultado


# In[145]:


resultado = conjunto1.issuperset(subconjunto)
resultado


# In[146]:


diff_simetrica = conjunto1.symmetric_difference(conjunto2)
diff_simetrica


# In[147]:


A = {1, 2, 3, 4}
B = {3, 4, 5, 6}

# União
print(A | B)  # {1, 2, 3, 4, 5, 6}

# Interseção
print(A & B)  # {3, 4}

# Diferença
print(A - B)  # {1, 2}

# Diferença Simétrica
print(A ^ B)  # {1, 2, 5, 6}


# In[148]:


5 in A


# In[150]:


for i in (A | B):
    print(i)


# In[151]:


A[0]


# In[152]:


A


# In[153]:


A_list = list(A)


# In[154]:


A_list[0]


# In[155]:


3 in A


# In[156]:


A_list.count(3)


# In[157]:


A_list.index(3)


# In[158]:


A_list[2]


# In[165]:


A=[[1,2,3], [4,5,6], [7, 8, 9]]

#printa as linhas da matriz A
for i in A:
    print(i)
print("\n")

A_t=[[k[0] for k in A], [k[1] for k in A], [k[2] for k in A]]

#printa as linhas da transposta de A
for i in A_t:
    print(i)


# In[180]:


A=[[1,2], [4,5], [7, 8]]

#printa as linhas da matriz A
for i in A:
    print(i)
print("\n")

A_t=[[k[0] for k in A], [k[1] for k in A]]

#printa as linhas da transposta de A
for i in A_t:
    print(i)


# In[181]:


import numpy as np

A = np.array([[1,2], [4,5], [7, 8]])

A_t = np.transpose(A)

print(A)
print("\n")
print(A_t)


# In[ ]:





# In[ ]:




