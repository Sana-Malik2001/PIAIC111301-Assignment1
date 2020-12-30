{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# %load E:/Assignment2_PIAIC.py\n",
    "# Read Instructions carefully before attempting this assignment\n",
    "\n",
    "# 1) don't rename any function name\n",
    "# 2) don't rename any variable name\n",
    "# 3) don't remove any #comment \n",
    "# 4) don't remove \"\"\" under triple quate values \"\"\"\n",
    "# 5) you have to write code where you found \"write your code here\"\n",
    "# 6) after download rename this file with this format \"PIAICCompletRollNumber_AssignmentNo.py\"\n",
    "#   Example piaic17896_Assignment1.py\n",
    "# 7) After complete this assignment please push on your own GitHub repository.\n",
    "# 8) you can submit this assignment through the google form\n",
    "# 9) copy this file absolute URL then paste in the google form\n",
    "#  The example above: https://github.com/EnggQasim/Batch04_to_35/blob/main/Sunday/1_30%20to%203_30/Assignments/assignment1.txt\n",
    "\n",
    "# * Because all assignment we will be checked through software if you missed any above points \n",
    "# * then we can't assign your scores in our database."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Task no 1\n",
    "def function1():\n",
    "    # create 2d array from 1,12 range \n",
    "    # dimension should be 6row 2 columns  \n",
    "    # and assign this array values in x values in x variable\n",
    "    # Hint: you can use arange and reshape numpy methods  \n",
    "    x =  np.arange(1,13).reshape((6,2)) \n",
    "\n",
    "    return x\n",
    "    \"\"\"\n",
    "    expected output:\n",
    "    [[ 1  2]\n",
    "    [ 3  4]\n",
    "    [ 5  6]\n",
    "    [ 7  8]\n",
    "    [ 9 10]\n",
    "    [11 12]]\n",
    "    \"\"\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "  x =  np.arange(1,13).reshape((6,2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1,  2],\n",
       "       [ 3,  4],\n",
       "       [ 5,  6],\n",
       "       [ 7,  8],\n",
       "       [ 9, 10],\n",
       "       [11, 12]])"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Task2\n",
    "def function2():\n",
    "    #create 3D array (3,3,3)\n",
    "    #must data type should have float64\n",
    "    #array value should be satart from 10 and end with 36 (both included)\n",
    "    # Hint: dtype, reshape \n",
    "    \n",
    "    x = np.arange(1,28,dtype=np.float64).reshape((3,3,3))    \n",
    "    x = np.arange(10,37,dtype=np.float64).reshape((3,3,3))\n",
    "\n",
    "    return x\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = np.arange(10,37,dtype=np.float64).reshape((3,3,3))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[10., 11., 12.],\n",
       "        [13., 14., 15.],\n",
       "        [16., 17., 18.]],\n",
       "\n",
       "       [[19., 20., 21.],\n",
       "        [22., 23., 24.],\n",
       "        [25., 26., 27.]],\n",
       "\n",
       "       [[28., 29., 30.],\n",
       "        [31., 32., 33.],\n",
       "        [34., 35., 36.]]])"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "#task3\n",
    "def function3():\n",
    "    #extract those numbers from given array. those are must exist in 5,7 Table\n",
    "    #example [35,70,105,..]\n",
    "    a = np.arange(1, 100*10+1).reshape((100,10))\n",
    "    x = a[(a%5==0) & (a%7==0)] \n",
    "    return x\n",
    "   "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "a=np.arange(1, 100*10+1).reshape((100,10))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "x=a[(a%5==0) & (a%7==0)]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 35,  70, 105, 140, 175, 210, 245, 280, 315, 350, 385, 420, 455,\n",
       "       490, 525, 560, 595, 630, 665, 700, 735, 770, 805, 840, 875, 910,\n",
       "       945, 980])"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "#task4 done\n",
    "def function4():\n",
    "    #Swap columns 1 and 2 in the array arr.\n",
    "   \n",
    "    arr = np.arange(9).reshape(3,3)\n",
    "  \n",
    "    return arr[:,[1,0,2]] \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "arr = np.arange(9).reshape(3,3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0, 1, 2],\n",
       "       [3, 4, 5],\n",
       "       [6, 7, 8]])"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "    \n",
    "#task5\n",
    "def function5():\n",
    "    #Create a null vector of size 20 with 4 rows and 5 columns with numpy function\n",
    "   \n",
    "    z = np.zeros(20).reshape(4,5)\n",
    "  \n",
    "    return z\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0., 0., 0., 0., 0.],\n",
       "       [0., 0., 0., 0., 0.],\n",
       "       [0., 0., 0., 0., 0.],\n",
       "       [0., 0., 0., 0., 0.]])"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.zeros(20).reshape(4,5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "#task6\n",
    "def function6():\n",
    "    # Create a null vector of size 10 but the fifth and eighth value which is 10,20 respectively\n",
    "   \n",
    "    arr=np.zeros(10) \n",
    "    arr[5]=10\n",
    "    arr[8]=20\n",
    "  \n",
    "    return arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "arr=np.zeros(10) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "arr[5]=10\n",
    "arr[8]=20"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0.,  0.,  0.,  0.,  0., 10.,  0.,  0., 20.,  0.])"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "   \n",
    "#task7\n",
    "def function7():\n",
    "    #  Create an array of zeros with the same shape and type as X. Dont use reshape method\n",
    "    x = np.arange(4, dtype=np.int64)\n",
    "  \n",
    "    return np.arange(4, dtype=np.int64)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    " x = np.arange(4, dtype=np.int64)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 2, 3], dtype=int64)"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "#task8\n",
    "def function8():\n",
    "    # Create a new array of 2x5 uints, filled with 6.\n",
    "    \n",
    "    x =np.full((2,5),6)\n",
    "    return x\n",
    "\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    " x =np.full((2,5),6)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[6, 6, 6, 6, 6],\n",
       "       [6, 6, 6, 6, 6]])"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "    \n",
    "#task9\n",
    "def function9():\n",
    "    # Create an array of 2, 4, 6, 8, ..., 100.\n",
    "    \n",
    "    a = np.arange(2,101,2)\n",
    "  \n",
    "    return a\n",
    "\n",
    "     "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "a = np.arange(2,101,2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([  2,   4,   6,   8,  10,  12,  14,  16,  18,  20,  22,  24,  26,\n",
       "        28,  30,  32,  34,  36,  38,  40,  42,  44,  46,  48,  50,  52,\n",
       "        54,  56,  58,  60,  62,  64,  66,  68,  70,  72,  74,  76,  78,\n",
       "        80,  82,  84,  86,  88,  90,  92,  94,  96,  98, 100])"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "    \n",
    "#task10\n",
    "def function10():\n",
    "    # Subtract the 1d array brr from the 2d array arr, such that each item of brr subtracts from respective row of arr.\n",
    "    \n",
    "    arr = np.array([[3,3,3],[4,4,4],[5,5,5]])\n",
    "    brr = np.array([1,2,3])\n",
    "    subt = arr-brr[:,None]\n",
    "  \n",
    "    return subt\n",
    "\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "arr = np.array([[3,3,3],[4,4,4],[5,5,5]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[3, 3, 3],\n",
       "       [4, 4, 4],\n",
       "       [5, 5, 5]])"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    " brr = np.array([1,2,3])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 2, 3])"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "brr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    " subt = arr-brr[:,None]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[2, 2, 2],\n",
       "       [2, 2, 2],\n",
       "       [2, 2, 2]])"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "subt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [],
   "source": [
    "    \n",
    "#task11\n",
    "def function11():\n",
    "    # Replace all odd numbers in arr with -1 without changing arr.\n",
    "    \n",
    "    arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])\n",
    "    ans = np.where(arr%2!=0,-1,arr)\n",
    "  \n",
    "    return ans\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    " ans = np.where(arr%2!=0,-1,arr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[-1, -1, -1],\n",
       "       [ 4,  4,  4],\n",
       "       [-1, -1, -1]])"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ans"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [],
   "source": [
    "#task12 done\n",
    "def function12():\n",
    "    # Create the following pattern without hardcoding. Use only numpy functions and the below input array arr.\n",
    "    # HINT: use stacking concept\n",
    "    \n",
    "    arr = np.array([1,2,3])\n",
    "    ans = np.r_[np.repeat(arr,3),np.tile(arr,3)]\n",
    "  \n",
    "    return ans\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [],
   "source": [
    "#task13\n",
    "def function13():\n",
    "    # Set a condition which gets all items between 5 and 10 from arr.\n",
    "    \n",
    "    \n",
    "    arr = np.array([2, 6, 1, 9, 10, 3, 27])\n",
    "    ans = arr[(arr>5) & (arr<10)]\n",
    "    return ans\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [],
   "source": [
    " ans = arr[(arr>5) & (arr<10)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([], dtype=int32)"
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ans"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [],
   "source": [
    "#task14\n",
    "def function14():\n",
    "    # Create an 8X3 integer array from a range between 10 to 34 such that the difference between each element is 1 and then Split the array into four equal-sized sub-arrays.\n",
    "    # Hint use split method\n",
    "    \n",
    "    \n",
    "    arr = numpy.arange(10, 34, 1).reshape((8,3))\n",
    "    ans = arr.reshape(8,3)\n",
    "    ans = np.split(ans,[1,3,5,7])\n",
    "  \n",
    "    return ans\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [],
   "source": [
    "#task15 \n",
    "def function15():\n",
    "    #Sort following NumPy array by the second column\n",
    "    \n",
    "    \n",
    "    arr = np.array([[ 8,  2, -2],[-4,  1,  7],[ 6,  3,  9]])\n",
    "    ans = arr[np.argsort(arr[:, 1])] \n",
    "  \n",
    "    return ans\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [],
   "source": [
    "ans = arr[np.argsort(arr[:, 1])]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[3, 3, 3],\n",
       "       [4, 4, 4],\n",
       "       [5, 5, 5]])"
      ]
     },
     "execution_count": 71,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ans"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [],
   "source": [
    "  #task16\n",
    "def function16():\n",
    "    #Write a NumPy program to join a sequence of arrays along depth.\n",
    "    \n",
    "    \n",
    "    x = np.array([[1], [2], [3]])\n",
    "    y = np.array([[2], [3], [4]])\n",
    "    ans = np.dstack((x,y))\n",
    "    return ans  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = np.array([[1], [2], [3]])\n",
    "y = np.array([[2], [3], [4]])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [],
   "source": [
    "ans = np.dstack((x,y))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[1, 2]],\n",
       "\n",
       "       [[2, 3]],\n",
       "\n",
       "       [[3, 4]]])"
      ]
     },
     "execution_count": 76,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ans"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [],
   "source": [
    "    \n",
    "#Task17 \n",
    "def function17():\n",
    "    # replace numbers with \"YES\" if it divided by 3 and 5\n",
    "    # otherwise it will be replaced with \"NO\"\n",
    "    # Hint: np.where\n",
    "    arr = np.arange(1,10*10+1).reshape((10,10))\n",
    "    return  np.where([(arr%3 == 0) & (arr%5 == 0)], \"YES\", \"NO\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [],
   "source": [
    " arr = np.arange(1,10*10+1).reshape((10,10))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[  1,   2,   3,   4,   5,   6,   7,   8,   9,  10],\n",
       "       [ 11,  12,  13,  14,  15,  16,  17,  18,  19,  20],\n",
       "       [ 21,  22,  23,  24,  25,  26,  27,  28,  29,  30],\n",
       "       [ 31,  32,  33,  34,  35,  36,  37,  38,  39,  40],\n",
       "       [ 41,  42,  43,  44,  45,  46,  47,  48,  49,  50],\n",
       "       [ 51,  52,  53,  54,  55,  56,  57,  58,  59,  60],\n",
       "       [ 61,  62,  63,  64,  65,  66,  67,  68,  69,  70],\n",
       "       [ 71,  72,  73,  74,  75,  76,  77,  78,  79,  80],\n",
       "       [ 81,  82,  83,  84,  85,  86,  87,  88,  89,  90],\n",
       "       [ 91,  92,  93,  94,  95,  96,  97,  98,  99, 100]])"
      ]
     },
     "execution_count": 82,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO'],\n",
       "        ['NO', 'NO', 'NO', 'NO', 'YES', 'NO', 'NO', 'NO', 'NO', 'NO'],\n",
       "        ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'YES'],\n",
       "        ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO'],\n",
       "        ['NO', 'NO', 'NO', 'NO', 'YES', 'NO', 'NO', 'NO', 'NO', 'NO'],\n",
       "        ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'YES'],\n",
       "        ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO'],\n",
       "        ['NO', 'NO', 'NO', 'NO', 'YES', 'NO', 'NO', 'NO', 'NO', 'NO'],\n",
       "        ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'YES'],\n",
       "        ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO']]],\n",
       "      dtype='<U3')"
      ]
     },
     "execution_count": 80,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.where([(arr%3 == 0) & (arr%5 == 0)], \"YES\", \"NO\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Task18\n",
    "def function18():\n",
    "    # count values of \"students\" are exist in \"piaic\"\n",
    "    piaic = np.arange(100)\n",
    "    students = np.array([5,20,50,200,301,7001])\n",
    "    x=np.count_nonzero(np.where(students<101))+1 \n",
    "    return x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {},
   "outputs": [],
   "source": [
    "piaic = np.arange(100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
       "       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n",
       "       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,\n",
       "       51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,\n",
       "       68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,\n",
       "       85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99])"
      ]
     },
     "execution_count": 86,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "piaic"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "metadata": {},
   "outputs": [],
   "source": [
    "students = np.array([5,20,50,200,301,7001])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([   5,   20,   50,  200,  301, 7001])"
      ]
     },
     "execution_count": 87,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "students"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [],
   "source": [
    "x=np.count_nonzero(np.where(students<101))+1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3"
      ]
     },
     "execution_count": 89,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Task19\n",
    "def function19():\n",
    "    #Create variable \"X\" from 1,25 (both are included) range values\n",
    "    #Convert \"X\" variable dimension into 5 rows and 5 columns\n",
    "    #Create one more variable \"W\" copy of \"X\" \n",
    "    #Swap \"W\" row and column axis (like transpose)\n",
    "    # then create variable \"b\" with value equal to 5\n",
    "    # Now return output as \"(X*W)+b:\n",
    "    X =  np.arange(1,26).reshape(5,5)\n",
    "    W =   np.copy(x)\n",
    "    W.T\n",
    "    b =np.array(5)\n",
    "    output =(x*W)+b "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "metadata": {},
   "outputs": [],
   "source": [
    "x=np.arange(1,26).reshape(5,5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1,  2,  3,  4,  5],\n",
       "       [ 6,  7,  8,  9, 10],\n",
       "       [11, 12, 13, 14, 15],\n",
       "       [16, 17, 18, 19, 20],\n",
       "       [21, 22, 23, 24, 25]])"
      ]
     },
     "execution_count": 92,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "metadata": {},
   "outputs": [],
   "source": [
    " W =   np.copy(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1,  2,  3,  4,  5],\n",
       "       [ 6,  7,  8,  9, 10],\n",
       "       [11, 12, 13, 14, 15],\n",
       "       [16, 17, 18, 19, 20],\n",
       "       [21, 22, 23, 24, 25]])"
      ]
     },
     "execution_count": 94,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "W"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1,  6, 11, 16, 21],\n",
       "       [ 2,  7, 12, 17, 22],\n",
       "       [ 3,  8, 13, 18, 23],\n",
       "       [ 4,  9, 14, 19, 24],\n",
       "       [ 5, 10, 15, 20, 25]])"
      ]
     },
     "execution_count": 95,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "W.T"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "metadata": {},
   "outputs": [],
   "source": [
    "b =np.array(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(5)"
      ]
     },
     "execution_count": 97,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "metadata": {},
   "outputs": [],
   "source": [
    "  output =(x*W)+b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[  6,   9,  14,  21,  30],\n",
       "       [ 41,  54,  69,  86, 105],\n",
       "       [126, 149, 174, 201, 230],\n",
       "       [261, 294, 329, 366, 405],\n",
       "       [446, 489, 534, 581, 630]])"
      ]
     },
     "execution_count": 99,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "output"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Task20\n",
    "def fucntion20():\n",
    "    #apply fuction \"abc\" on each value of Array \"X\"\n",
    "    x = np.arange(1,11)\n",
    "    def xyz(x):\n",
    "        return x*2+3-2\n",
    "\n",
    "    return x*2+3-2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "metadata": {},
   "outputs": [],
   "source": [
    " x = np.arange(1,11)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 3,  5,  7,  9, 11, 13, 15, 17, 19, 21])"
      ]
     },
     "execution_count": 102,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x*2+3-2"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
