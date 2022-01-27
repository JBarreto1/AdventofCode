# Function list for Advent of code! adventofcode.com
# I'm a bit behind - starting on dec. 12

from numpy import zeros,count_nonzero
from collections import defaultdict,Counter
from operator import itemgetter
from heapq import heappop, heappush
from copy import deepcopy
from sys import maxsize
from math import sqrt
from itertools import product,combinations,chain,combinations_with_replacement
from ast import literal_eval
import functools
# import matplotlib.pyplot as plt
import copy

#day 1
# #how many depth increases are there in the list?
def depthIncreaseCounter(ls):
    depthInc = 0
    for i in range(1,len(ls)):
        if ls[i] > ls[i-1]:
            depthInc += 1
    return depthInc

#group depth increases by sliding window.
def groupCounter(ls):
    newLs = []
    for i in range(0, len(ls)-2):
        newItem = ls[i] + ls[i+1] + ls[i+2] #window size is 3
        newLs.append(newItem)
    return newLs

#day 2
# Given the list of instructions for the submarine, what is the depth and position?
#instr expected format: [['up','1'],['forward','4']] (direction and magnitude in list of tuple list)
def finalPosition(instr):
    horiz = 0
    depth = 0
    for i in range(0,len(instr)):
        if instr[i][0] == 'forward':
            horiz += int(instr[i][1])
        if instr[i][0] == 'down':
            depth += int(instr[i][1])
        if instr[i][0] == 'up':
            depth -= int(instr[i][1])
    return [horiz,depth]

def finalWithAim(instr):
    horiz = 0
    depth = 0
    aim = 0
    for i in range(0,len(instr)):
        if instr[i][0] == 'forward':
            horiz += int(instr[i][1])
            depth += aim * int(instr[i][1])
        if instr[i][0] == 'down':
            aim += int(instr[i][1])
        if instr[i][0] == 'up':
            aim -= int(instr[i][1])
    return [horiz,depth]

#day 3
def Gamma(ls):
    ans = ''
    for i in range(0,len(ls[0])):
        zeros = 0
        for number in ls:
            if number[i] == '0':
                zeros += 1
        if zeros > len(ls) / 2:
            ans += '0'
        else:
            ans += '1'
    return ans

def eps(binStr):
    ans = ''
    for x in binStr:
        if x == '1':
            ans += '0'
        else:
            ans += '1'
    return ans

def oxygen(ls):
    valueLen = len(ls[0])
    for i in range(0, valueLen):
        zeros = []
        ones = []
        for value in ls:
            if value[i] == '0':
                zeros.append(value)
            else:
                ones.append(value)
        if len(zeros) > len(ones):
            ls = zeros.copy()
        else:
            ls = ones.copy()
        if len(ls) == 1:
            return ls[0]

def co2(ls):
    valueLen = len(ls[0])
    for i in range(0, valueLen):
        zeros = []
        ones = []
        for value in ls:
            if value[i] == '0':
                zeros.append(value)
            else:
                ones.append(value)
        if len(zeros) > len(ones):
            ls = ones.copy()
        else:
            ls = zeros.copy()
        if len(ls) == 1:
            return ls[0]

#Day 4
def bingo(nums, boards):
    for k in range(0, len(nums)):
        for board in boards:
            searchRes = searchABoard(board,nums[k])
            if searchRes != 'notFound':
                i,j = searchRes
                board[i][j] = 'X'
                if hasBingo(board):
                    return sumBoard(board)*nums[k]

def hasBingo(board):
    for i in range(0,len(board)):
        tokens = 0
        for j in range(0, len(board[i])):
            if board[i][j] == 'X':
                tokens += 1
        if tokens == 5: #hardcode to 5 because its a gauranteed 5x5 board
            return True
    for j in range(0,len(board[j])):
        tokens = 0
        for i in range(0, len(board[j])):
            if board[i][j] == 'X':
                tokens += 1
        if tokens == 5:
            return True
    return False

def searchABoard(board,num):
    for i in range(0,5):
        for j in range(0,5):
            if board[i][j] == num:
                return (i,j)
    return 'notFound'

def processBingo(data):
    nums = [int(x) for x in data[0][:-1].split(',')]
    boards = []
    for i in range(2,len(data),6):
        board = []
        for j in range(0,5):
            line = [int(x) for x in data[i+j][:-1].split(' ') if x != '']
            # line = data[i+j][:-1].split(' ')
            board.append(line)
        boards.append(board)
    return (boards,nums)

def sumBoard(board):
    ans = 0
    for i in range(0,5):
        for j in range(0,5):
            if board[i][j] != 'X':
                ans += board[i][j]
    return ans

def lastBingoBoard(nums,boards):
    totalBoards = len(boards)
    for k in range(0, len(nums)):
        for key,board in enumerate(boards):
            if board != 'bingo':
                searchRes = searchABoard(board,nums[k])
                if searchRes != 'notFound':
                    i,j = searchRes
                    board[i][j] = 'X'
                    if totalBoards == 1:
                        if hasBingo(board):
                            return sumBoard(board)*nums[k]
                    else:
                        if hasBingo(board):
                            boards[key] = 'bingo'
                            totalBoards -= 1


#day 5 Vents
def processVents(data,noDiag):
    #segments listed in list of tuples - format: [(x1,y1),(x2,y2)]
    segments = []
    maxX = 0
    maxY = 0
    for line in data:
        line = line[:-1].split(' -> ')
        first = line[0].split(',')
        sec = line[1].split(',')
        maxX = max([int(first[0]),int(sec[0]),maxX])
        maxY = max([int(first[1]),int(sec[1]),maxY])
        coord = [(int(first[0]),int(first[1])),(int(sec[0]),int(sec[1]))]
        if noDiag:
            if int(first[0]) == int(sec[0]) or int(first[1]) == int(sec[1]):
                segments.append(coord)
        else:
            segments.append(coord)
    return segments,maxX,maxY

def ventMap(segments,X,Y):
    vents = zeros((X+1,Y+1))
    for seg in segments:
        x1,y1 = seg[0]
        x2,y2 = seg[1]
        starty = min(y1,y2)
        endy = max(y1,y2)
        startx = min(x1,x2)
        endx = max(x1,x2)
        if x1 == x2 or y1 == y2:
            for i in range(starty,endy+1):
                for j in range(startx,endx+1):
                    vents[i][j] += 1
        else:
            steps = endy - starty
            for i in range(0,steps+1):
                if (x1 - x2 > 0 and y1 - y2 > 0) or (x1 - x2 < 0 and y1 - y2 < 0):
                    vents[starty+i][startx+i] += 1
                else:
                    vents[starty+i][endx-i] += 1
    return vents


def ventUnsafeCount(vents):
    ans = 0
    for i in range(0,len(vents)):
        for j in range(0, len(vents)):
            if vents[i][j] > 1:
                ans += 1
    return ans

#Day 6
def processFish(data):
    return [int(x) for x in data[0].split(',')]

#this turned out to be to SLOW for round two (256 days)
def fishdayCount(fishList,days):
    for i in range(0,days):
        fishList = nextDay(fishList)
        counts = []
        for j in range(0,9):
            counts.append(fishList.count(j))
        print(str(i) + ': ', len(fishList),' ',counts)
        # print(str(i) + ': ', len(fishList),' ',fishList)
    return len(fishList)

def nextDay(fishList):
    addFish = 0
    ans = []
    for key,fish in enumerate(fishList):
        if fish == 0:
            addFish += 1
            fishList[key] = 6
        else:
            fishList[key] -= 1
    for i in range(0,addFish):
        fishList.append(8)
    return fishList

def buildDict(data,days):
    ans = 0
    for i in range(1,6):
        ans += fishdayCount([i],days)*data.count(i)
    return ans

#Another attempt, this time with recursion - same time complexity as the first attempt, still too slow
def fishRecurs(fish):
    if fish == 0:
        return 6,6
    else:
        return fish - 1

def fishListRecurs(fishList,days):
    if days == 0:
        return fishList
    else:
        ans = [6, 8] * fishList.count(0)
        # print(ans)
        for i in range(1,9):
            ans = ans + ([i - 1] * fishList.count(i))
            # print(ans)
        # print(ans)
        # if days % 10 == 0:
        #     print(days)
        return fishListRecurs(ans,days-1)

#Final attempt - keep a count of the numers and slide them over - this works for a very large number of days!
def fishMovingList(fishList,days):
    counter = []
    for i in range(0,9):
        counter.append(fishList.count(i))
        # print(fishList.count(i))
    for i in range(1,days+1):
        counter = moveCounter(counter)
        # print(i, ": ",sum(counter)," ",counter)
    return sum(counter)

def moveCounter(fishCounter):
    ans = fishCounter[1:]
    parent = fishCounter[0]
    ans[6] += parent
    ans.append(parent)
    return ans

#Day 7
def processHoriz(data):
    return [int(x) for x in data[0].split(',')]

def cheapestHoriz(ls):
    ls.sort()
    return ls[round(len(ls)/2) - 1]

def fuelCalc(ls):
    center = cheapestHoriz(ls)
    ans = 0
    for pos in ls:
        ans += abs(pos - center)
    return ans

def variableCenter(ls):
    #Cheated here a bit - I thought the answer would be the average, but it is apparently not always. Instead I calcualted the fuel for points around (10 before and after average) and used the smallest one
    return round(sum(ls)/len(ls))

def variableFuelCalc(ls,center):
    ans = 0
    for pos in ls:
        for i in range(0,abs(pos - center)+1):
            ans += i
    return ans

#day 8
def processingOutputVal(data):
    ans = []
    for line in data:
        intVal = line.split('|')[1][:-1]
        outVal = intVal.split(' ')[1:]
        ans.append(outVal)
    return ans

def procInputVal(data):
    ans = []
    for line in data:
        intVal = line.split('|')[0]
        outVal = intVal.split(' ')[:-1]
        ans.append(outVal)
    return ans

def uniqueSegNumbers(ls):
    uniqueNums = 0
    for outVal in ls:
        for segs in outVal:
            #unique values 2->1, 7 -> 8, 3-> 7, 4 -> 4
            if len(segs) in [2,7,3,4]:
                uniqueNums += 1
    return uniqueNums

def idSegUniq(ls):
    ans = []
    for segs in ls:
        if len(segs) == 2:
            ans.append()

def createSegDict(ls):
    #unique values 2->1, 7 -> 8, 3-> 7, 4 -> 4
    ans = [0] * 10
    ans[1] = set([segs for segs in ls if len(segs) == 2][0])
    ans[4] = set([segs for segs in ls if len(segs) == 4][0])
    ans[7] = set([segs for segs in ls if len(segs) == 3][0])
    ans[8] = set([segs for segs in ls if len(segs) == 7][0])
    four = set(ans[4])
    one = set(ans[1])
    difference = list(four.difference(one))
    sixSegs = [segs for segs in ls if len(segs) == 6]
    for num in sixSegs:
        # print(sixSegs)
        #ZERO is the one that doesn't have both of the segs in the difference between 1 and 4
        if difference[0] not in num or difference[1] not in num:
            ans[0] = set(num)
        #SIX is the one that doesn't have both of the segs in 1
        elif list(one)[0] not in num or list(one)[1] not in num:
            ans[6] = set(num)
        #NINE is the only one remaining in the sixSegs group
        else:
            ans[9] = set(num)
    fiveSegs = [segs for segs in ls if len(segs) == 5]
    for num in fiveSegs:
        #THREE has both of the 1 segments
        if list(one)[0] in num and list(one)[1] in num:
            ans[3] = set(num)
        #FIVE has both of the 4/1 difference segments
        elif difference[0] in num and difference[1] in num:
            ans[5] = set(num)
        else:
            ans[2] = set(num)
    return ans

def evalauteOutVal(digits,outVal):
    segDict = createSegDict(digits)
    print(segDict)
    print(outVal)
    outStr = ''
    for val in outVal:
        outStr += str(segDict.index(set(val)))
    return outStr

def evalAllSets(digits,outVal):
    ans = 0
    for i in range(0,len(outVal)):
        # print(digits[i])
        num = int(evalauteOutVal(digits[i],outVal[i]))
        print(num)
        ans += num
    return ans

#Day 9
def processHieghtMap(data):
    ans = [[9] * (len(data[0])+1)]
    for line in data:
        lineStr = [9]
        lineStr += [int(x) for x in line[:-1]]
        lineStr.append(9)
        ans.append(lineStr)
    ans.append([9] * (len(data[0])+1))
    return ans

def lowPoints(heightMap):
    ans = 0
    for i in range(1,len(heightMap)-1):
        for j in range(1,len(heightMap[1])-1):
            print(i,' ',j,' ',heightMap[i][j])
            if heightMap[i][j] < min([heightMap[i][j+1],heightMap[i][j-1],heightMap[i+1][j],heightMap[i-1][j]]):
                ans += heightMap[i][j] + 1
                print('add: ',heightMap[i][j])
    return ans

def basinSearch(heightMap,coord):
    i, j = coord
    heightMap[i][j] = 'done'
    basinSize = 1
    if heightMap[i][j+1] != 9 and heightMap[i][j+1] != 'done':
        heightMap,bSize = basinSearch(heightMap,(i,j+1))
        basinSize += bSize
    if heightMap[i][j-1] != 9 and heightMap[i][j-1] != 'done':
        heightMap,bSize = basinSearch(heightMap,(i,j-1))
        basinSize += bSize
    if heightMap[i+1][j] != 9 and heightMap[i+1][j] != 'done':
        heightMap,bSize = basinSearch(heightMap,(i+1,j))
        basinSize += bSize
    if heightMap[i-1][j] != 9 and heightMap[i-1][j] != 'done':
        heightMap,bSize = basinSearch(heightMap,(i-1,j))
        basinSize += bSize
    return heightMap,basinSize
    # while keepSearch:
    #     if heightMap[i][j+1] != 9:
    #         basinSize += 1
    #     if heightMap[i][j-1]
    #     if heightMap[i+1][j]
    #     if heightMap[i-1][j]

def findBasin(heightMap):
    basinList = []
    for i in range(1,len(heightMap)-1):
        for j in range(1,len(heightMap[1])-1):
            # print(i,' ',j,' ',heightMap[i][j])
            if heightMap[i][j] != 9 and heightMap[i][j] != 'done':
                heighMap,basinSize = basinSearch(heightMap,(i,j))
                basinList.append(basinSize)
                # print(heighMap)
                # print(basinSize)
    basinList.sort(reverse=True)
    ans = 1
    for i in range(0,3):
        ans *= basinList[i]
    return ans
             
#Day 10
def processSynError(data):
    return [line[:-1] for line in data]

def corruptFinder(data):
    points = {
        ")": 3,
        "]": 57,
        "}": 1197,
        ">": 25137
    }
    incompPoints = {
        "(": 1,
        "[": 2,
        "{": 3,
        "<": 4
    }
    coor = {
        ")": '(',
        "]": '[',
        "}": '{',
        ">": '<'
    }
    corruptScore = 0
    incompScores = []
    for line in data:
        i = 0
        notCorrupt = True
        record = []
        while notCorrupt and i < len(line):
            # print(line[i],record)
            if line[i] in ['(','{','[','<']:
                record.append(line[i])
                i += 1
            elif coor[line[i]] == record[-1]:
                record.pop()
                i += 1
            else:
                # print(i,points[line[i]])
                corruptScore += points[line[i]]
                notCorrupt = False
        incompScore = 0
        if notCorrupt:
            for char in reversed(record):
                incompScore *= 5
                incompScore += incompPoints[char]
            incompScores.append(incompScore)
    incompScores.sort()
    # print(incompScores)
    ansIncomp = incompScores[int(len(incompScores)/2)]
    return corruptScore,ansIncomp

#Day 11
def processOctopi(data):
    ans = [[0] * (len(data[0])+1)]
    for line in data:
        lineStr = [0]
        lineStr += [int(x) for x in line[:-1]]
        lineStr.append(0)
        ans.append(lineStr)
    ans.append([0] * (len(data[0])+1))
    return ans

def flashCount(octopi):
    flashes = 0
    # for _ in range(0,100):
    notDone = True
    for i in range(1,len(octopi)-1):
        for j in range(1,len(octopi[1])-1):
            octopi[i][j] += 1
    while notDone:
        for i in range(1,len(octopi)-1):
            for j in range(1,len(octopi[1])-1):
                # print(i,j)
                if octopi[i][j] != -1:
                    if octopi[i][j] > 9:
                        for k in range(-1,2):
                            for l in range(-1,2):
                                # print(i,j,k,l)
                                if octopi[i+k][j+l] != -1:
                                    octopi[i+k][j+l] += 1
                        octopi[i][j] = -1
                        flashes += 1
        # print(any([any([j > 9 for j in i]) for i in octopi]))
        if not any([any([j > 9 for j in i[1:-1]]) for i in octopi[1:-1]]):
            octopi = [[0 if j == -1 else j for j in i] for i in octopi]
            notDone = False
    return flashes,octopi

def runFlash(octopi,num):
    ans = 0
    for i in range(num):
        count,octopi = flashCount(octopi)
        print(octopi)
        ans += count
    return ans

def firstSimultaneous(octopi):
    ans = 0
    notAll = True
    while notAll:
        _,octopi = flashCount(octopi)
        print(octopi)
        if all([all([j == 0 for j in i[1:-1]]) for i in octopi[1:-1]]):
            notAll = False
        ans += 1
    return ans

#Day 12
def processPath(data):
    ans = [[line.split('-')[0],line[:-1].split('-')[1]] for line in data]
    return [[y,x] if x == 'end' or y == 'start' else [x,y] for x,y in ans]

def findPath(pathLs):
    paths = list(filter(lambda path: path[0] == 'start',pathLs))
    connectLS = list(filter(lambda path: path[0] != 'start',pathLs))
    # allPaths = []
    # for path in pathLs:
    #     if path[0] == 'start':
    #         allPaths.append(path)
    donePath = []
    i = 0
    notDone = True
    while i<3:
        for path in paths:
            print(path)
            for j in connectLS:
                # print(j)
                beg,stop = j
                if beg == path[-1]:
                    if not (stop.islower() and stop in path):
                        add = path.copy()
                        add.append(stop)
                        if stop == 'end':
                            donePath.append(add)
                        else:
                            paths.append(add)
                elif stop == path[-1]:
                    if not (beg.islower() and beg in path):
                        add = path.copy()
                        add.append(beg)
                        if beg == 'end':
                            donePath.append(add)
                        paths.append(add)
        print(i)
        i += 1
    return donePath


def DFSglo(pathLS):
    neighbors = neighDICT(pathLS)
    visited = set()
    # visited = [0] * len(neighbors.keys())
    # visited = visDict(neighbors.keys())
    count = [0]
    # for key in neighbors.keys():
    #     visited = visDict(neighbors.keys())
    #     if key.islower() and key not in ['start','end']:
    #         visited[key] = -1
    #         print(visited)
    #         dfs(['start'],visited,neighbors,count)
    dfs([],'start',visited,neighbors,count,True)
    # for key in neighbors.keys():
    #     if key.islower() and key not in ['start','end']:
    #         tempDict = copy.deepcopy(neighbors)
    #         new = key + 'a'
    #         tempDict[new] = neighbors[key]
    #         for neigh in neighbors[key]:
    #             tempDict[neigh].append(new)
    #         print(tempDict)
    #         dfs(['start'],visited,tempDict,count)
    return count

def visDict(nodes):
    vis = {}
    for key in nodes:
        vis[key] = 0
    return vis


def neighDICT(pathLS):
    neighbors = defaultdict(list)
    for connect in pathLS:
        neighbors[connect[0]].append(connect[1])
        neighbors[connect[1]].append(connect[0])
    return neighbors

def dfs(path,node,visited,neighbors,count,revisited):
    if node in path and node.islower():
        # visited[node[-1]] += 1
        # if node in visited:
        #     revisited = False
        revisited = False
        # visited.add(node)
    # visited.add(node)
    path.append(node)

    if node == 'end':
        print(path)
        count[0] += 1
    else:
        for neighbor in neighbors[node]:
            
            # if neighbor.isupper() or neighbor not in visited:
            # if revisited == True:
            #     valid = neighbor not in ['start','end']
            # else:
            #     valid = neighbor.isupper() or neighbor not in visited
            # if valid:
            #     dfs(path,node,visited,neighbors,count,revisited)
            if revisited == True:
                valid = neighbor != 'start'
            else:
                # valid = neighbor.isupper() or neighbor not in visited:
                valid = neighbor.isupper() or not neighbor in path
            if valid:
                dfs(path,neighbor,visited,neighbors,count,revisited)
    # if node.islower():
        # visited[node[-1]] -= 1
        # visited.remove(node)
    path.pop()

#DAY 13
def procTransPaper(data):
    lines = []
    dots = set()
    num = data.index('\n')
    for line in data[:num]:
        inter = line[:-1].split(',')
        ans = (int(inter[0]),int(inter[1]))
        dots.add(ans)
    # print(dots)
    for line in data[num+1:]:
        lines.append([line[11],int(line[13:])])
        # if line[11] == 'y':
        #     lines.append((0,int(line[13:])))
        # else:
        #     lines.append((int(line[13:]),0))
    return dots,lines
    # print(lines)

def fold(dots,line):
    if line[0] == 'y':
        dotLs = list(filter(lambda dot: dot[1] > line[1],dots))
        for dot in dotLs:
            newCoord = (line[1] * 2) - dot[1]
            ans = (dot[0],newCoord)
            # print(dot[0],newCoord)
            dots.remove(dot)
            dots.add(ans)
    else:
        dotLs = list(filter(lambda dot: dot[0] > line[1],dots))
        for dot in dotLs:
            newCoord = (line[1] * 2) - dot[0]
            ans = (newCoord,dot[1])
            dots.remove(dot)
            dots.add(ans)
    # print(dots)
    return dots

def allFolds(dots,lines):
    for line in lines:
        dots = fold(dots,line)
    return dots

def visualizeDots(dots):
    # ans = list(dots)
    # res = list(zip(*ans))
    # plt.figure(figsize=(50,50))
    # plt.scatter(res[0],res[1])
    # plt.show()
    maxcoord = max(dots,key=itemgetter(1))[0]
    for i in range(maxcoord,-1,-1):
        print('')
        for j in range(maxcoord):
            if (i,j) in dots:
                print('X',end='')
            else:
                print(' ',end='')

#Day 14
def procPolymer(data):
    start = data[0][:-1]
    template = dict()
    for line in data[2:]:
        ans = line.split(' -> ')
        template[ans[0]] = ans[1][:-1]
    return start,template

def oneStepPoly(start,polyDict):
    insertions = []
    for i in range(len(start)-1):
        # print(start[i:i+2])
        if start[i:i+2] in polyDict.keys():
            insertions.append((i,polyDict[start[i:i+2]]))
    # print(insertions)
    return insertPoly(start,insertions)

def insertPoly(start,insertions):
    insertions.reverse()
    for insert in insertions:
        i,letter = insert
        start = start[:i+1] + letter + start[i+1:]
    return start

def polyBuild(start,polyDict,steps):
    for i in range(steps):
        print(i)
        start = oneStepPoly(start,polyDict)
    counts = Counter(start).most_common()
    most = counts[0][1]
    least = counts[-1][1]
    # print(most,least)
    # print(Counter(start).most_common()[-1])
    return most-least

def initPolyCounter(start):
    polyCounter = dict()
    for i in range(len(start)-1):
        # print(start[i:i+2])
        if start[i:i+2] in polyCounter.keys():
            polyCounter[start[i:i+2]] += 1
        else:
            polyCounter[start[i:i+2]] = 1
    return polyCounter

def onePolyCounter(polyCounter,polyDict):
    ansDict = dict()
    for key in polyCounter.keys():
        ans = key[0] + polyDict[key] + key[1]
        for i in range(2):
            if ans[i:i+2] in ansDict.keys():
                ansDict[ans[i:i+2]] += polyCounter[key]
            else:
                ansDict[ans[i:i+2]] = polyCounter[key]
    return ansDict

def polyCounter(start,polyDict,steps):
    polyCounter = initPolyCounter(start)
    for i in range(steps):
        print(i,polyCounter)
        polyCounter = onePolyCounter(polyCounter,polyDict)
    finalCount = dict()
    for _,(key,val) in enumerate(polyCounter.items()):
        # for i in range(2):
        if key[0] in finalCount.keys():
            finalCount[key[0]] += val
        else:
            finalCount[key[0]] = val
    if start[-1] in finalCount.keys():
        finalCount[start[-1]] += 1
    else:
        finalCount[start[-1]] = 1   
    most = max(finalCount, key=finalCount.get)
    least = min(finalCount, key=finalCount.get)
    print(most,least)
    return finalCount[most]-finalCount[least]

#Day 15
def processCaveHole(data):
    ans = []
    for line in data:
        ans.append([int(i) for i in line[:-1]])
    return ans

def findPath(cave):
    visited = set()
    maxJ = len(cave[0]) - 1
    maxI = len(cave) - 1
    unvisited = set((i,j) for i in range(maxI+1) for j in range(maxJ+1))
    graph = graphDict(maxI,maxJ)
    current_node = (0,0)
    graph[(current_node[0] , current_node[1])] = (0,'')
    calculatePaths(cave,current_node,visited,unvisited,graph,maxI,maxJ)
    print(graph)
    return graph[(maxI,maxJ)]

def graphDict(maxI,maxJ):
    graph = dict()
    initValues = maxsize
    for j in range(maxJ+1):
        for i in range(maxI+1):
            graph[(i,j)] = [initValues,'']
    return graph

def neighbors(node,cave,maxI,maxJ):
    neighbors = []
    i,j = node
    # if i - 1 >= 0:
    #     neighbors.append((cave[i - 1][j] , i - 1 , j))
    # if i + 1 <= maxI:
    #     neighbors.append((cave[i + 1][j] , i + 1 , j))
    # if j - 1 >= 0:
    #     neighbors.append((cave[i][j - 1] , i , j - 1))
    # if j + 1 <= maxJ:
    #     neighbors.append((cave[i][j + 1] , i , j + 1))
    if i - 1 >= 0:
        neighbors.append((i - 1 , j))
    if i + 1 <= maxI:
        neighbors.append((i + 1 , j))
    if j - 1 >= 0:
        neighbors.append((i , j - 1))
    if j + 1 <= maxJ:
        neighbors.append((i , j + 1))
    return neighbors

def calculatePaths(cave,current_node,visited,unvisited,graph,maxI,maxJ):
    notDone = True
    while notDone:
        neighborhood = neighbors(current_node,cave,maxI,maxJ)
        # neighborhood = filter(lambda x: not x in visited,)
        # neighborhood.sort(key=lambda x: x[0])
        for neighbor in neighborhood:
            i,j = neighbor
            val = graph[current_node][0] + cave[i][j]
            if val < graph[(i,j)][0]:
                # print(i,j,val , graph[current_node[0]][current_node[1]][0])
                graph[(i,j)] = (val, current_node)
        visited.add(current_node)
        unvisited.remove(current_node)
        print(len(visited))
        if unvisited != set():
            current_node = min(list(filter(lambda k: not k in visited, graph.keys())), key=(lambda k: graph[k][0]))
        else:
            notDone = False
        # if unvisited != set():
            #Regular Dijkstra's algorithm says check the next node with the smallest distance
            # current_node = min(list(filter(lambda k: not k in visited, graph.keys())), key=(lambda k: graph[k][0]))
            #A* uses a heuristic to determine the next node to check
            # current_node = min(list(filter(lambda k: not k in visited, graph.keys())), key=(lambda k: graph[k][0]))
            # print(current_node,graph)
            # calculatePaths(cave,current_node,visited,unvisited,graph,maxI,maxJ)


def findPathHEAPED(cave):
    visited = []
    maxJ = len(cave[0]) - 1
    maxI = len(cave) - 1
    graph = graphDict(maxI,maxJ)
    current_node = (0,0)
    unvisited = [(heuristic(current_node,maxI,maxJ) + cave[0][0] , cave[0][0] , current_node)]
    return calculatePathsHEAPED(cave,current_node,visited,unvisited,graph,maxI,maxJ)

def heuristic(node,maxI,maxJ):
    i,j = node
    return maxI - i + maxJ - j

def getNeighbours(x,y,maxI,maxJ):
    if x > 0: yield(x-1,y)
    if x < maxI: yield(x+1,y)
    if y > 0: yield(x,y-1)
    if y < maxJ: yield(x,y+1)

def calculatePathsHEAPED(cave,current_node,visited,unvisited,graph,maxI,maxJ):
    loop = 0
    testSet = set()
    while unvisited and loop < 100:
        #unvisited contains the heuristic distance and node
        estimator,risk,current_node = heappop(unvisited)
        testSet.add(current_node)
        if current_node == (maxI,maxJ):
            temp = [x for x in visited if x not in testSet]
            print('temp: ',temp)
            print(loop)
            return graph[current_node][0] - cave[0][0]
        if not current_node in visited:
            neighborhood = neighbors(current_node,cave,maxI,maxJ)
            for neighbor in neighborhood:
                if not neighbor in visited:
                    i,j = neighbor
                    # if i<20 and j<20:
                    #     print(i,j)
                    val = risk + cave[i][j]
                    if graph[neighbor][0] > val:
                        graph[neighbor] = val,current_node
                        priority = val + heuristic(neighbor,maxI,maxJ)
                        heappush(unvisited,(priority,val,neighbor))
        loop += 1     
        visited.append(current_node)

#restart attempt 3!!!
def findPathGraphed(cave):
    visited = set()
    maxJ = len(cave[0]) - 1
    maxI = len(cave) - 1
    graph = dict()
    current_node = (0,0)
    graph[current_node] = heuristic(current_node,maxI,maxJ)+cave[0][0],cave[0][0]
    return calculatePathsGRAPHED(cave,current_node,visited,graph,maxI,maxJ)

def calculatePathsGRAPHED(cave,current_node,visited,graph,maxI,maxJ):
    loop = 0
    while (maxI,maxJ) not in visited:
        if loop % 1000 == 0:
            print(loop)
        current_node = min(graph, key=lambda x: graph[x][0])
        estimate,dist = graph[current_node]
        visited.add(current_node)
        del(graph[current_node])
        if current_node == (maxI,maxJ):
            print(loop)
            return dist - cave[0][0]
        neighborhood = neighbors(current_node,cave,maxI,maxJ)
        for neighbor in neighborhood:
            if not neighbor in visited:
                i,j = neighbor
                val = dist + cave[i][j]
                newDist = min(graph.get(neighbor,(0,maxsize))[1],val)
                graph[neighbor] = (newDist+heuristic(neighbor,maxI,maxJ),newDist)
        loop += 1     

def expandCave(cave):
    ans = deepcopy(cave)
    for j in range(len(ans)):
        for i in range(1,5):
            ans[j] += [x + i if x + i < 10 else x + i - 9 for x in cave[j]]
    for i in range(1,5):
        for j in range(len(cave)):
            ans.append([x + i if x + i < 10 else x + i - 9 for x in ans[j]])
    return ans

#Day 16
def hexCodeDict():
    ans = {
        '0' : '0000',
        '1' : '0001',
        '2' : '0010',
        '3' : '0011',
        '4' : '0100',
        '5' : '0101',
        '6' : '0110',
        '7' : '0111',
        '8' : '1000',
        '9' : '1001',
        'A' : '1010',
        'B' : '1011',
        'C' : '1100',
        'D' : '1101',
        'E' : '1110',
        'F' : '1111',
    }
    return ans

def hextobin(hex):
    ans = ''
    hexDict = hexCodeDict()
    for dig in hex:
        ans += hexDict[dig]
    return ans

def binDecode(bin_code):
    # print(bin_code)
    pack_vers = int(bin_code[0:3],2)
    pack_type = int(bin_code[3:6],2)
    if pack_type == 4:
        bin_ans = ''
        code_length = 0
        not_done = True
        i = 6
        while not_done:
            bin_ans += bin_code[i+1:i+5]
            if bin_code[i] == '0':
                not_done = False
            i += 5
        # print("version",pack_vers,bin_ans)
        return bin_ans,i,pack_vers
    else:
        int_ans = []
        length_type = int(bin_code[6],2)
        if length_type == 0:
            #15 bits
            total_length = int(bin_code[7:22],2)
            i = 22
            while i - 22 < total_length:
                # print(i,bin_code[i:22+total_length])
                sub_pack,code_length,pack_v = binDecode(bin_code[i:22+total_length])
                pack_vers += pack_v
                # print(code_length,total_length,int(sub_pack,2))
                
                i += code_length
                # print(i-22,total_length)
                int_ans.append(int(sub_pack,2))
        else:
            #11 bits
            total_length = int(bin_code[7:18],2)
            i = 18
            for _ in range(total_length):
                sub_pack,code_length,pack_v = binDecode(bin_code[i:])
                pack_vers += pack_v
                i += code_length
                int_ans.append(int(sub_pack,2))
            # print("version",pack_vers,int_ans)
        int_ans = ansByType(int_ans,pack_type)
        return bin(int_ans).replace("0b",""),i,pack_vers

def ansByType(int_ans,pack_type):
    if pack_type == 0:
        res = sum(int_ans)
    elif pack_type == 1:
        res = 1
        for x in int_ans:
            res *= x
    elif pack_type == 2:
        res = min(int_ans)
    elif pack_type == 3:
        res = max(int_ans)
    elif pack_type == 5:
        if int_ans[0] > int_ans[1]:
            res = 1
        else:
            res = 0
    elif pack_type == 6:
        if int_ans[0] < int_ans[1]:
            res = 1
        else:
            res = 0
    elif pack_type == 7:
        if int_ans[0] == int_ans[1]:
            res = 1
        else:
            res = 0
    return res

#day 17
# def minX()

def yHit(change,window):
    minWin,maxWin = window
    pos = 0
    ans = []
    while pos >= minWin:
        pos += change
        change -= 1
        if pos >= minWin and pos <= maxWin:
            ans.append(1)
        else:
            ans.append(0)
    return ans

def xHit(x_change,window,cycles):
    minWin,maxWin = window
    pos = 0
    ans = []
    loop = 0
    while loop < cycles and pos < maxWin:
        pos += x_change
        if x_change != 0:
            x_change -= 1
        if pos >= minWin and pos <= maxWin:
            ans.append(1)
        else:
            ans.append(0)
        loop += 1
    return ans

def maxVert(maxY):
    # for i in range(10)
    # print(vertWindow(9,(-10,-5)))
    count = 0
    for i in range(maxY):
        count += i
    return count


#x has to be at least (x+1)*x/2 and at most maxX
#y has to be at least -maxY and at most maxY

def trajectory_count(x_window,y_window):
    minX,maxX = x_window
    minY,maxY = y_window
    # start_x = int((minX+1)*minX/2)
    start_x = int((sqrt(4*(minX*2+(1/4)))-1)/2)
    count = []
    for x in range(start_x,maxX+1):
        for y in range(minY,1-minY):
            y_list = yHit(y,y_window)
            x_list = xHit(x,x_window,len(y_list))
            print(x,y,list(zip(x_list,y_list)))
            if (1,1) in list(zip(x_list,y_list)):
                count.append((x,y))
    return count

#Day 18
def proccessSnailNums(data):
    ans = []
    for line in data:
        ans.append(literal_eval(line[:-1]))
    return ans


def addSnailNums(a,b):
    return [a,b]

def reduceSnailNums(num):
    # print(num)
    exploded = explodeSnail(num)
    if exploded:
        return reduceSnailNums(exploded)
    split = splitSnail(num)
    if split:
        return reduceSnailNums(split)
    # print(num)
    return num

def explodeSnail(num):
    recent_num = False
    next_num = 0
    ans = []
    right_val = 0
    left_val = 0
    right = False
    for indA,a in enumerate(num):
        if isinstance(a,int):
            if right:
                # if recent_num:
                #     num = changeLeft(num,recent_num,left_value)
                num[indA] += right_val
                return num
                # next_num.append((indA))
            else:
                # recent_num.append((indA))
                recent_num = [indA]
        else:
            for indB,b in enumerate(a):
                if isinstance(b,int):
                    if right:
                        # if recent_num:
                        #     num = changeLeft(num,recent_num,left_value)
                        num[indA][indB] += right_val
                        return num
                        # next_num.append((indA,indB))
                    else:
                        # recent_num.append((indA,indB))
                        recent_num = [indA,indB]
                else:
                    for indC,c in enumerate(b):
                        if isinstance(c,int):
                            if right:
                                # if recent_num:
                                #     num = changeLeft(num,recent_num,left_value)
                                num[indA][indB][indC] += right_val
                                return num
                                # next_num.append((indA,indB,indC))
                            else:
                                # recent_num.append((indA,indB,indC))
                                recent_num = [indA,indB,indC]
                        else:
                            for indD,d in enumerate(c):
                                if isinstance(d,int):
                                    if right:
                                        # if recent_num:
                                        #     num = changeLeft(num,recent_num,left_value)
                                        num[indA][indB][indC][indD] += right_val
                                        return num
                                        # next_num.append((indA,indB,indC,indD))
                                    else:
                                        # recent_num.append((indA,indB,indC,indD))
                                        recent_num = [indA,indB,indC,indD]
                                else:
                                    # ans = [indA,indB,indC,indD]
                                    # print(d)
                                    if right:
                                        # print(num,indA,indB,indC,indD)
                                        num[indA][indB][indC][indD][0] += right_val
                                        return num
                                    else:
                                        left_val,right_val = d
                                        if recent_num:
                                            num = changeLeft(num,recent_num,left_val)
                                        num[indA][indB][indC][indD] = 0
                                        right = True
                                    
    if right:
        return num    
    else:
        return False

def changeLeft(num,recent_num,left_val):
    if len(recent_num) == 1:
        num[recent_num[0]] += left_val
    elif len(recent_num) == 2:
        num[recent_num[0]][recent_num[1]] += left_val
    elif len(recent_num) == 3:
        num[recent_num[0]][recent_num[1]][recent_num[2]] += left_val
    elif len(recent_num) == 4:
        num[recent_num[0]][recent_num[1]][recent_num[2]][recent_num[3]] += left_val
    return num

# test = [[[[0, [3, 2]], [3, 3]], [4, 4]], [5, 5]]
# print(test)
# print(explodeSnail(test))

def splitSnail(num):
    for indA,a in enumerate(num):
        # print(indA)
        if type(a) == int:
            if a > 9:
                new = int(a/2)
                if a % 2 == 0:
                    num[indA] = [new,new]
                else:
                    num[indA] = [new,new+1]
                return num
        else:
            for indB,b in enumerate(a):
                # print(indA,indB)
                if type(b) == int:
                    if b > 9:
                        new = int(b/2)
                        if b % 2 == 0:
                            num[indA][indB] = [new,new]
                        else:
                            num[indA][indB] = [new,new+1]
                        return num
                else:
                    for indC,c in enumerate(b):
                        # print(indA,indB,indC)
                        if type(c) == int:
                            if c > 9:
                                new = int(c/2)
                                if c % 2 == 0:
                                    num[indA][indB][indC] = [new,new]
                                else:
                                    num[indA][indB][indC] = [new,new+1]
                                return num
                        else:
                            for indD,d in enumerate(c):
                                # print(indA,indB,indC)
                                if d > 9:
                                    new = int(d/2)
                                    if d % 2 == 0:
                                        num[indA][indB][indC][indD] = [new,new]
                                    else:
                                        num[indA][indB][indC][indD] = [new,new+1]
                                    return num

    return False

def magnitude(num):
    # print(num)
    # if type(num[0]) == int and type(num[1]) == int:
    #     return num[0] * 3 + num[1] * 2
    if type(num[0]) == int:
        if type(num[1]) == int:
            # print('both')
            return num[0] * 3 + num[1] * 2
        else:
            # print('first is in second is not')
            return num[0] * 3 + magnitude(num[1]) * 2
    elif type(num[1]) == int:
        # print('here')
        return magnitude(num[0]) * 3 + num[1] * 2
    else:
        # print('there')
        return magnitude(num[0]) * 3 + magnitude(num[1]) * 2

# test = [[[[6,6],[7,6]],[[7,7],[7,0]]],[[[7,7],[7,7]],[[7,8],[9,9]]]]
# print(test)
# print(magnitude(test))

# a = [[2,[[7,7],7]],[[5,8],[[9,3],[0,2]]]]
# b = [[[0,[5,8]],[[1,7],[9,6]]],[[4,[1,2]],[[1,4],2]]]
# red = reduceSnailNums(addSnailNums(a,b))
# print(red)
# print(magnitude(red))


#Day 19
def proccessScanners(data):
    ans = []
    scanner = []
    for line in data:
        if line != '\n':
            if 'scanner' in line:
                # print(line[:-1])
                ans.append(scanner)
                scanner = []
            else:
                coords = line[:-1].split(',')
                # print(coords)
                scanner.append((int(coords[0]),int(coords[1]),int(coords[2])))
    ans.append(scanner)
    return ans

def beaconDist(scanner):
    vectors = defaultdict(set)
    for i in range(len(scanner)):
        for j in range(i+1,len(scanner)):
            vectors[scanner[i]].add((scanner[i][0]-scanner[j][0],scanner[i][1]-scanner[j][1],scanner[i][2]-scanner[j][2]))
            vectors[scanner[j]].add((scanner[j][0]-scanner[i][0],scanner[j][1]-scanner[i][1],scanner[j][2]-scanner[i][2]))
            # res.setdefault(scanner[i],[]).append((scanner[i][0]-scanner[j][0],scanner[i][1]-scanner[j][1],scanner[i][2]-scanner[j][2]))
            # res.setdefault(scanner[j],[]).append((scanner[j][0]-scanner[i][0],scanner[j][1]-scanner[i][1],scanner[j][2]-scanner[i][2]))
            # res[i,j] = (scanner[i][0]-scanner[j][0],scanner[i][1]-scanner[j][1],scanner[i][2]-scanner[j][2])
    return vectors

def tuplesMatch(tup_a,tup_b):
    ans = []
    for i in tup_a:
        if i in tup_b:
            ans.append(1)
        elif -i in tup_b:
            ans.append(-1)
    if len(ans) == 3:
        return ans
    else:
        return False

    # a_set = set()
    # for i in tup_a:
    #     a_set.add(i)
    #     a_set.add(-i)
    # b_set = set()
    # for j in tup_b:
    #     b_set.add(j)
    #     b_set.add(-j)
    # return a_set == b_set

def calcMoreDist(scan_a,beacon,scan_dict):
    for i in range(len(scan_a)):
        scan_dict[i,len(scan_a)] = (scanner[i][0]-scanner[j][0],scanner[i][1]-scanner[j][1],scanner[i][2]-scanner[j][2])

def findTranslator(point_a,point_b):
    return (point_a[0] + point_b[0], point_a[1] + point_b[1], point_a[2] + point_b[2])

def translatorDict(paired_points,scan_a,scan_b):
    ans = dict()
    for i in range(len(paired_points)):
        ans[paired_points[i][0][0],paired_points[i][1][0]] = findTranslator(scan_a[paired_points[i][0][0]],scan_b[paired_points[i][1][0]])
        ans[paired_points[i][0][1],paired_points[i][1][1]] = findTranslator(scan_a[paired_points[i][0][1]],scan_b[paired_points[i][1][1]])
        ans[paired_points[i][0][0],paired_points[i][1][1]] = findTranslator(scan_a[paired_points[i][0][0]],scan_b[paired_points[i][1][1]])
        ans[paired_points[i][0][1],paired_points[i][1][0]] = findTranslator(scan_a[paired_points[i][0][1]],scan_b[paired_points[i][1][0]])
    return ans

def compareDist(scan_a,scan_b):
    # print(beaconDist(scan_a))
    dist_a = beaconDist(scan_a)
    dist_b = beaconDist(scan_b)
    # ans = dict()
    key_a,val_a = list(dist_a.items())[0]
    firstPairs = []
    matches = set()
    trans = []
    translations = []
    new_beacons = set()
    for key_a,val_a in dist_a.items():
        for key_b,val_b in dist_b.items():
            matchRes = tuplesMatch(val_a,val_b)
            if matchRes:
                # print(key_a,matchRes)
                translations.append([key_a,key_b])
                # print(val_a)
                matches.add(key_b[0])
                matches.add(key_b[1])
                if firstPairs == []:
                    firstPairs = [key_a,key_b]
                elif trans == []:
                    # print(list(set(firstPairs[1]).intersection(key_b))[0])
                    # print(scan_a[key_a[0]])
                    # print(key_b,firstPairs[1],scan_b[list(set(firstPairs[1]).intersection(key_b))[0]])
                    trans = findTranslator(scan_a[key_a[0]],scan_b[list(set(firstPairs[1]).intersection(key_b))[0]])
                    # print(key_a,key_b)
            # else:
            #     new_beacons.add(key_b)
            # ans[key_a[0]]
    print(trans)
    trans_dict = translatorDict(translations,scan_a,scan_b)
    for key in trans_dict.keys():
        print(key,trans_dict[key])
    # print(ans)
    # print(trans)
    # for u in ans:
    #     # print(scan_b[u])
    #     print(trans[0]-scan_b[u][0],trans[1]-scan_b[u][1],trans[2]-scan_b[u][2])
    # print(ans)

def rotate(scan_a,i):
    def rotations(beacon,i):
        x,y,z = beacon
        rotates = [(x, y, z),
                    (z, y, -x),
                    (-x, y, -z),
                    (-z, y, x),
                    (-x, -y, z),
                    (-z, -y, -x),
                    (x, -y, -z),
                    (z, -y, x),
                    (x, -z, y),
                    (y, -z, -x),
                    (-x, -z, -y),
                    (-y, -z, x),
                    (x, z, -y),
                    (-y, z, -x),
                    (-x, z, y),
                    (y, z, x),
                    (z, x, y),
                    (y, x, -z),
                    (-z, x, -y),
                    (-y, x, z),
                    (-z, -x, y),
                    (y, -x, z),
                    (z, -x, -y),
                    (-y, -x, -z)]
        return rotates[i]
    new_beacons = [rotations(beacon, i) for beacon in scan_a]
    return new_beacons

def find_overlaps(scan_a,scan_b):
    a_vectors = beaconDist(scan_a)
    b_vectors = beaconDist(scan_b)
    overlaps = defaultdict(set)
    beacon_pairs = product(scan_a,scan_b)
    for a_beacon, b_beacon in beacon_pairs:
        a_vect = a_vectors[a_beacon]
        b_vect = b_vectors[b_beacon]
        # print('a',a_vect,'b',b_vect)
        overlap = len(a_vect & b_vect)
        if overlap >= 11:
            overlaps[a_beacon].add(b_beacon)
            # print(overlaps)
    return overlaps

def find_and_normalize(scanners):
    orig = deepcopy(scanners)
    anchor = scanners.pop(0)
    scanner_coords = {}
    ROTATIONS = 24
    

    while scanners:
        tested_scanner = scanners.pop(0)
        offset = False
        overlap_found = False
        print(len(scanners))
        for i in range(ROTATIONS):
            # print(i)
            rotated_scanner = tested_scanner.copy()
            rotated_scanner = rotate(rotated_scanner,i)
            overlaps = find_overlaps(anchor, rotated_scanner)
            if overlaps:
                overlap_found = True
                a_beacon = list(overlaps.keys())[0]
                b_beacon = overlaps[a_beacon].pop()
                offset = (a_beacon[0]-b_beacon[0], a_beacon[1]-b_beacon[1], a_beacon[2]-b_beacon[2])
                
                scanner_coords[orig.index(tested_scanner)] = offset

                new_beacons = []
                for beacon in rotated_scanner:
                    x, y, z = beacon
                    ox, oy, oz = offset
                    offset_beacon = (x + ox, y + oy, z + oz)
                    new_beacons.append(offset_beacon)
                anchor += new_beacons
                break

        if not overlap_found:
            scanners.append(tested_scanner)

    return len(set(anchor)), scanner_coords

#algorithm isn't great - it takes five minutes to complete. I kept the offsets and I'll simply paste them into scanner_coords variable here to avoid running the script again.
scanner_coords = {38: (147, 155, -1150), 15: (-1111, -15, -1172), 17: (84, -12, -2408), 35: (-1069, 1334, -1335), 4: (152, 57, -3678), 5: (-1125, 2470, -1239), 10: (140, 1336, -2439), 18: (-2300, 2420, -1172), 26: (-1061, 3690, -1300), 31: (-1172, 1211, -2375), 33: (91, 1367, -3627), 34: (37, 3619, -1177), 36: (25, -1081, -3562), 2: (-2296, 1210, -2351), 6: (73, 2415, -2429), 7: (-1041, 3750, -2376), 8: (-1166, 2471, -3551), 14: (86, 4852, -1290), 16: (-1168, 4811, -2507), 19: (51, 6081, -1202), 22: (40, -2227, -3679), 24: (139, 7342, -1144), 27: (-1201, 6159, -1249), 28: (61, 1318, -4788), 29: (130, 2535, -3661), 32: (4, 6024, -123), 39: (-1091, 4964, -52), 1: (39, -3432, -3735), 3: (-1230, 1218, -4909), 11: (-2393, 4828, -81), 12: (10, 4959, 1), 13: (-1133, -2226, -3620), 20: (43, -2315, -2539), 21: (1189, 1325, -4821), 23: (-7, 2465, -4812), 25: (1176, 1237, -5981), 30: (1182, 2551, -6086), 37: (30, 2407, -5973), 9: (-2428, -2405, -3634)}

def manhattan_dist(point_a,point_b):
    x1,y1,z1 = point_a
    x2,y2,z2 = point_b
    return abs(x1-x2) + abs(y1-y2) + abs(z1-z2)

def farthest_dist(scanner_coords):
    pairs = combinations(scanner_coords,2)
    biggest = 0
    for i,j in pairs:
        print(i,j)
        next_dist = manhattan_dist(scanner_coords[i],scanner_coords[j])
        if next_dist > biggest:
            biggest = next_dist
    return biggest

# print(farthest_dist(scanner_coords))

#Day 20
def process_depth(data):
    enhance_algo = data.pop(0)[:-1].replace('#','1').replace('.','0')
    input_img = []
    for line in data[1:]:
        input_img.append(line[:-1].replace('#','1').replace('.','0'))
    return input_img, enhance_algo

# def input_window(location,input_img):
#     x,y = location
#     for i in range(3):
#         for j in range(3):

def add_border(input_img,border_size,all_on):
    token = '0'
    if all_on:
        token = '1'
    starter = token * (len(input_img)+border_size*2)
    ans = []
    for _ in range(border_size):
        ans.append(starter)
    for line in input_img:
        ans.append((token*border_size)+line+(token*border_size))
    for _ in range(border_size):
        ans.append(starter)
    return ans

def enhance_img(input_img,enhance_algo,all_on):
    border_size = 2
    if enhance_algo[0] == '1':
        border_size = 10
    img_bordered = add_border(input_img,border_size,all_on)
    if enhance_algo[0] == '1':
        all_on = not all_on
    ans = []
    count = 0
    # img_bordered.replace('#','1')
    # img_bordered.replace('.','0')
    for i in range(1,len(img_bordered)-1):
        line = ''
        for j in range(1,len(img_bordered[0])-1):
            window = ''
            for k in range(-1,2):
                window += img_bordered[i+k][j-1:j+2]
            # print(int(window,2))
            line += enhance_algo[int(window,2)]
        ans.append(line)
        count += line.count('1')
    return ans, count, all_on
            
def print_enhanced(inp_img):
    for line in inp_img:
        print(line.replace('0','.').replace('1','#'))          

def apply_enhancement(input_img,enhance_algo,num):
    all_on = False
    for i in range(num):
        input_img,count,all_on = enhance_img(input_img,enhance_algo,all_on)
        print(i)
        # print_enhanced(input_img)
    return count

#day 21
def roll(pos1,pos2,max_score):
    num = -1
    ans1 = 0
    ans2 = 0
    rolls = 0
    while ans1 < max_score and ans2 < max_score:
        if rolls % 2 == 0:
            ans1,pos1,num = turn(ans1,pos1,num)
        else:
            ans2,pos2,num = turn(ans2,pos2,num)
        rolls += 1
    return rolls * 3 * min(ans1,ans2)

def turn(ans,pos,num):
    num += 3
    if (num * 3 + pos) % 10 == 0:
        pos = 10
    else:
        pos = (num * 3 + pos) % 10
    ans += pos
    return ans,pos,num

def take_turn(pos,turn_val):
    if (turn_val + pos) % 10 == 0:
        pos = 10
    else:
        pos = (turn_val + pos) % 10
    return pos

# @functools.lru_cache(maxsize=None)
def tree_gamed2(pos1,ans1,pos2,ans2,memo_dict,true_value_dict):
    w1 = w2 = 0
    if (pos1,ans1,pos2,ans2) in memo_dict.keys():
        return memo_dict[(pos1,ans1,pos2,ans2)]
    else:
        for turn_val in turn_value_dict.keys():
            new_pos = take_turn(pos1,turn_val)
            new_ans = ans1 + new_pos
            if new_ans > 20:
                w1 += 1 * turn_value_dict[turn_val]
            else:
                new_w2,new_w1 = tree_gamed2(pos2,ans2,new_pos,new_ans,memo_dict,true_value_dict)
                w1 += (new_w1 * turn_value_dict[turn_val])
                w2 += (new_w2 * turn_value_dict[turn_val])
        memo_dict[(pos1,ans1,pos2,ans2)] = w1,w2
        return w1,w2

#Day 22
def processCuboids(data):
    ans = []
    bound_x = [0,0]
    bound_y = [0,0]
    bound_z = [0,0]
    for line in data:
        if line[0:2] == 'on':
            coords = [1]
        else:
            coords = [0]
        text_coords = line[:-1].split(' ')[1].split(',')
        for i in range(3):
            start,end = text_coords[i].split('=')[1].split('..')
            start = int(start)
            end = int(end)
            if i == 0:
                if start < bound_x[0]:
                    bound_x[0] = start
                if end > bound_x[1]:
                    bound_x[1] = end
            if i == 1:
                if start < bound_y[0]:
                    bound_y[0] = start
                if end > bound_y[1]:
                    bound_y[1] = end
            if i == 2:
                if start < bound_z[0]:
                    bound_z[0] = start
                if end > bound_z[1]:
                    bound_z[1] = end
            coords.append((start,end))
        ans.append(coords)
    return bound_x,bound_y,bound_z,ans

def cuboid_switches_initial(instructions,n):
    total = zeros((n,n,n))
    # print(count_nonzero(total))
    for line in instructions:
        x1,x2 = line[1]
        y1,y2 = line[2]
        z1,z2 = line[3]
        if all(a <= 50 and a >= -50 for a in [x1,x2,y1,y2,z1,z2]):
            # print([x1,x2,y1,y2,z1,z2])
            for i in range(x1+50,x2+51):
                # print('x')
                for j in range(y1+50,y2+51):
                    # print('y')
                    for k in range(z1+50,z2+51):
                        total[i][j][k] = line[0]
        # else:
        #     print('not within',line)
        # print(line,count_nonzero(total))
    return count_nonzero(total)
    # return Counter(chain(total))

def cuboids_intersect(cuboid_1,cuboid_2):
    x_1 = cuboid_1[0]
    y_1 = cuboid_1[1]
    z_1 = cuboid_1[2]
    x_2 = cuboid_2[0]
    y_2 = cuboid_2[1]
    z_2 = cuboid_2[2]
    if x_1[0] > x_2[1] or x_1[1] < x_2[0]:
        return False
    if y_1[0] > y_2[1] or y_1[1] < y_2[0]:
        return False
    if z_1[0] > z_2[1] or z_1[1] < z_2[0]:
        return False
    return True

def trim_dim(x_1,x_2):
    if x_2[0] >= x_1[0]:
        if x_2[1] > x_1[1]:
            x_list = [(x_1[1]+1,x_2[1])],(x_2[0],x_1[1])
        else:
            x_list = [],(x_2[0],x_2[1])
    else:
        if x_2[1] > x_1[1]:
            x_list = [(x_2[0],x_1[0]-1),(x_1[1]+1,x_2[1])],(x_1[0],x_1[1])
        else:
            x_list = [(x_2[0],x_1[0]-1)],(x_1[0],x_2[1])
    return x_list

def smash(cuboid_1,cuboid_2):
    #1 is the established cuboid
    #2 is the cuboid I'm going to smash
    new_cuboids = []
    # for cube in filter(lambda x: not cuboids_intersect(x,cuboid_1), list(product(dims[0],dims[1],dims[2]))):
    #     new_cuboids.append([1,cube[0],cube[1],cube[2]])
    # new_cuboids = filter(lambda x: not cuboids_intersect(x,cuboid_1), list(product(dims[0],dims[1],dims[2])))
    #Trim the x direction
    trimmed,remaining = trim_dim(cuboid_1[0],cuboid_2[0])
    for cube in trimmed:
        new_cuboids.append([cube,cuboid_2[1],cuboid_2[2]])
        # print('new',new_cuboids)
    cuboid_2[0] = remaining
    # print('new',new_cuboids)
    # print('remaining',cuboid_2)
    #trim the y direction
    trimmed,remaining = trim_dim(cuboid_1[1],cuboid_2[1])
    for cube in trimmed:
        new_cuboids.append([cuboid_2[0],cube,cuboid_2[2]])
        # print('new',new_cuboids)
    cuboid_2[1] = remaining
    # print('new',new_cuboids)
    # print('remaining',cuboid_2)
    #trim the z direction
    trimmed,remaining = trim_dim(cuboid_1[2],cuboid_2[2])
    for cube in trimmed:
        new_cuboids.append([cuboid_2[0],cuboid_2[1],cube])
        # print('new',new_cuboids)
    # cuboid_2[2] = remaining
    # print('new',new_cuboids)
    # print('remaining',cuboid_2)
    return new_cuboids

def calc_cuboid_volume(cuboid):
    ans = 1
    for i in range(3):
        ans *= cuboid[i][1]+1-cuboid[i][0]
    return ans


def cuboid_switches_Total(instructions):
    core = [] #core contains the cuboids that are on
    while instructions:
        line = instructions.pop(0)
        current_cuboid = [line[1],line[2],line[3]]
        # print(current_cuboid)
        for cuboid in core:
            # print(current_cuboid,"core: ",cuboid)
            if cuboids_intersect(cuboid,current_cuboid):
                if line[0] == 1:
                    #SMASH current_cuboid and add sub-cuboids that don't intersect back to the front of the instructions list
                    #Go back to the beginning of the loop
                    new_cuboids = [[1,x,y,z] for (x,y,z) in smash(cuboid,current_cuboid)]
                    instructions = new_cuboids + instructions
                    break
                else:
                    #SMASH cuboid and add only the pieces that don't intersect with current_cuboid
                    # print(len(core))
                    core.remove(cuboid)
                    # print(len(core))
                    new_cuboids = smash(current_cuboid,cuboid)
                    core += new_cuboids
                    instructions.insert(0,line)
                    break
        else:
            if line[0] == 1:
                core.append(current_cuboid)
        #check that the current cuboid intersects any other cuboids in the core list
        #if the cuboid intersects and it's an 'on' instruction - SMASH the current cuboid and add only the cuboids
        #if the cuboid intersects and it's an 'off' instruction - SMASH the cuboid already in the core and remove the cuboid that should be off
    ans = 0
    for cube in core:
        ans += calc_cuboid_volume(cube)
        # print(cube,ans)
    return ans
    # return Counter(chain(total))
    
#Day 23
# def initialize_map(door_a,door_b,door_c,door_d):
#     hall = [0] * 11
#     connections = {
#         'a':2
#         'b':4
#         'c':6
#         'd':8
#     }
#     energy = {
#         'a':1
#         'b':10
#         'c':100
#         'd':1000
#     }

# def read_puzzle(data):
#     return ''.join([c for c in data if c in 'ABCD.'])

def read_puzzle(filename):
  with open(filename) as f:
    return ''.join([c for c in f.read() if c in 'ABCD.'])

def blocked(a,b,puzzle):
    step = 1 
    if a > b:
        step = -1
    for pos in range(a+step,b+step,step):
        if puzzle[pos] != '.':
            return True


def can_leave_room(puzzle,room_pos):
    for a in room_pos:
        if puzzle[a] == '.': continue
        return a

def get_possible_parc_pos(a,parc,puzzle):
    for b in [pos for pos in parc if puzzle[pos] == '.']:
        if blocked(a,b,puzzle): continue
        yield b

def move(a,b,puzzle):
    p = list(puzzle)
    p[b],p[a] = p[a],p[b]
    return ''.join(p)

def can_enter_room(a,b,amphi,puzzle,room_pos):
    for pos in room_pos:
        if puzzle[pos] == '.':
            best_pos = pos
        elif puzzle[pos] != amphi:
            return False
    if not blocked(a,b,puzzle):
        return best_pos

def possible_moves(puzzle, parc, stepout, target):
    for a in [pos for pos in parc if puzzle[pos] != '.']:
        amphi = puzzle[a]
        b = can_enter_room(a, stepout[amphi], amphi, puzzle, target[amphi])
        if b:
            yield a,b
    for room in 'ABCD':
        a = can_leave_room(puzzle, target[room])
        if not a: continue
        for b in get_possible_parc_pos(stepout[room], parc, puzzle):
            yield a,b

def solve(puzzle):
    energy = dict(A=1, B=10, C=100, D=1000)
    parc= [0,1,3,5,7,9,10]
    stepout = dict(A=2,B=4, C=6, D=8)
    target = {r: range(ord(r)-54,len(puzzle),4) for r in 'ABCD'}
    targetI = {v:key for key,val in target.items() for v in val}
    solution = '.'*11+'ABCD'*((len(puzzle)-11)//4)
    heap, seen = [(0,puzzle)], {puzzle:0}
    while heap:
        cost, state = heappop(heap)
        if state == solution: return cost
        for a,b in possible_moves(state, parc, stepout, target):
            p,r = (a,b) if a < b else (b,a)
            distance = abs(stepout[targetI[r]] - p) + (r-7)//4
            new_cost = cost + distance * energy[state[a]]
            moved = move(a,b,state)
            if seen.get(moved,999999) <= new_cost: continue
            seen[moved] = new_cost
            heappush(heap,(new_cost, moved))
    

#Day 25
def process_sea_cucumbers(data):
    ans = []
    for line in data:
        ans.append(list(line[:-1]))
    return ans

def east_step(sea_cucumber):
    ans = []
    for sea_cu in sea_cucumber:
        last = sea_cu[0]
        moving = []
        for i in range(len(sea_cu)-1,-1,-1):
            # print(i,last,sea_cu[i])
            if last == '.':
                if sea_cu[i] == '>':
                    moving.append(i)
            last = sea_cu[i]
        for move in moving:
            sea_cu[move] = '.'
            if move == len(sea_cu)-1:
                sea_cu[0] = '>'
            else:
                sea_cu[move+1] = '>'
        ans.append(sea_cu)
    return ans

# print(east_step(['.', '.', '.', '>', '>', '>', '.', '.', '>']))

def south_step(sea_cucumber):
    ans = [[] for i in range(len(sea_cucumber))]
    for j in range(0,len(sea_cucumber[0])):
        last = sea_cucumber[0][j]
        moving = []
        for i in range(len(sea_cucumber)-1,-1,-1):
            if last == '.':
                if sea_cucumber[i][j] == 'v':
                    moving.append(i)
            last = sea_cucumber[i][j]
        for move in moving:
            sea_cucumber[move][j] = '.'
            if move == len(sea_cucumber)-1:
                sea_cucumber[0][j] = 'v'
            else:
                sea_cucumber[move+1][j] = 'v'
        for k in range(len(sea_cucumber)):
            ans[k].append(sea_cucumber[k][j])
    return ans

# print(south_step([['.', '.', 'v'],['.', 'v', '>']]))

def solve_sea_cucumbers(sea_cucumbers):
    steps = 0
    last_step = []
    while sea_cucumbers != last_step:
        print(steps)
        last_step = deepcopy(sea_cucumbers)
        sea_cucumbers = deepcopy(east_step(sea_cucumbers))
        sea_cucumbers = deepcopy(south_step(sea_cucumbers))
        # print(print_cuc(sea_cucumbers))
        steps += 1
    return steps

#day 24
def process_instructions(data):
    ans = []
    for line in data:
        ans.append(line[:-1].split(' '))
    return ans

def check_digit(num):
    if num.isdigit():
        return True
    elif num[1:].isdigit():
        return True
    else:
        return False

def num_in_instruc(num,instruc_list,output_z):
    inp_counter = 0
    value_store = dict(w=0, x=0, y=0, z=output_z)
    for instruc in instruc_list:
        if instruc[0] == 'inp':
            value_store[instruc[1]] = int(num[inp_counter])
            inp_counter += 1
            # print(value_store)
        if instruc[0] == 'add':
            if check_digit(instruc[2]):
                value_store[instruc[1]] = value_store[instruc[1]] + int(instruc[2])
            else:
                value_store[instruc[1]] = value_store[instruc[1]] + value_store[instruc[2]]
        if instruc[0] == 'mul':
            if check_digit(instruc[2]):
                value_store[instruc[1]] = value_store[instruc[1]] * int(instruc[2])
            else:
                value_store[instruc[1]] = value_store[instruc[1]] * value_store[instruc[2]]
        if instruc[0] == 'div':
            if check_digit(instruc[2]):
                if int(instruc[2]) == 0:
                    return False
                else:
                    value_store[instruc[1]] = int(value_store[instruc[1]] / int(instruc[2]))
            else:
                if value_store[instruc[2]] == 0:
                    return False
                else:
                    value_store[instruc[1]] = int(value_store[instruc[1]] / value_store[instruc[2]])
        if instruc[0] == 'mod':
            if check_digit(instruc[2]):
                if int(instruc[2]) <= 0 or value_store[instruc[1]] < 0:
                    return False
                else:
                    value_store[instruc[1]] = value_store[instruc[1]] % int(instruc[2])
            else:
                if value_store[instruc[2]] <= 0 or value_store[instruc[1]] < 0:
                    return False
                else:
                    value_store[instruc[1]] = value_store[instruc[1]] % value_store[instruc[2]]
        if instruc[0] == 'eql':
            if check_digit(instruc[2]):
                if value_store[instruc[1]] == int(instruc[2]):
                    value_store[instruc[1]] = 1
                else:
                    value_store[instruc[1]] = 0
            else:
                if value_store[instruc[1]] == value_store[instruc[2]]:
                    value_store[instruc[1]] = 1
                else:
                    value_store[instruc[1]] = 0
        # print(instruc, value_store)
    # print(value_store)
    # if value_store['z'] == 0:
    #     return True
    # else:
    #     return False
    return value_store

def find_largest(instruc_list):
    # for i in range(99999999999999,99999999999199,-1):
    for i in range(9,0,-1):
        if '0' in str(i): continue
        num = list(str(i))
        # print(num,num_in_instruc(num,instruc_list))
        print(i,num_in_instruc(num,instruc_list))
        # if num_in_instruc(num,instruc_list):
        #     return i

def chunked(instruc_list):
    ans = []
    for chunk in range(0,len(instruc_list),18):
        # print(chunk,instruc_list[chunk-18:chunk])
        ans.append([(0,0),(0,0)])
        maxZ = 0
        minZ = 9999999
        if len(ans) == 1:
            for i in range(9,0,-1):
                num = list(str(i))
                output_z = num_in_instruc(num,instruc_list[chunk:chunk+18],0)['z']
                if output_z > maxZ:
                    maxZ = output_z
                    ans[-1][1] = (i,maxZ)
                if output_z < minZ:
                    minZ = output_z
                    ans[-1][0] = (i,minZ)
        else:
            for i in ans[-2]:
                num = [i[0]]
                output_z = num_in_instruc(num,instruc_list[chunk:chunk+18],i[1])['z']
                if output_z > maxZ:
                    maxZ = output_z
                    ans[-1][1] = (i[0],maxZ)
                if output_z < minZ:
                    minZ = output_z
                    ans[-1][0] = (i[0],minZ)
        print(ans)
        # print(i,num_in_instruc(num,instruc_list[chunk:chunk+18]))



def print_cuc(sea_cu):
    for line in sea_cu:
        print(''.join(line))

if __name__ == '__main__':
    test = 0
    if test == 1:
        with open ("test.txt", "r") as myfile:
            data=myfile.readlines()
    else:
        with open ("adventInput.txt", "r") as myfile:
            data=myfile.readlines()

    #PROCESSING DAY 24
    instruc_list = process_instructions(data)
    print(instruc_list)
    # print(num_in_instruc([5],instruc_list))
    # find_largest(instruc_list)
    chunked(instruc_list)
    # print(99999999999999-5520918017)
    # print(num_in_instruc(,instruc_list))
    # print(num_in_instruc(list(str(99994479081982-314125)),instruc_list))
    # print(num_in_instruc(list(str(99999999999999-5520918018)),instruc_list))
    # print(num_in_instruc(list(str(999),instruc_list))
    # print(num_in_instruc(list(str(999)),instruc_list)['z'])
    # print(num_in_instruc(list(str(999 - 12081 ),instruc_list))

    #PROCESSING DAY 25
    # sea_cucumbers = process_sea_cucumbers(data)
    # print(print_cuc(sea_cucumbers))
    # print(solve_sea_cucumbers(sea_cucumbers))
    # sea_cucumbers = deepcopy(east_step(sea_cucumbers))
    # print(print_cuc(sea_cucumbers))
    # sea_cucumbers = deepcopy(south_step(sea_cucumbers))
    # print(print_cuc(sea_cucumbers))
    # print(east_step(sea_cucumbers))
    # print(south_step(sea_cucumbers))



    #PROCESSING DAY 23
    # puzzle = read_puzzle('test.txt')
    # print(solve(puzzle))

    # PROCESS DAY 22
    # bound_x,bound_y,bound_z,instructions = processCuboids(data)
    # print(instructions)
    # # print(instructions)
    # # instructions = [[1,(10,12),(10,12),(10,12)]]
    # # print(cubiod_switches(instructions,101))
    # # bound_x,bound_y,bound_z,
    # print(cuboid_switches_Total(instructions))
    
    #PROCESS DAY 21
    
    # turn_value_dict = {
    #     3: 1,
    #     4: 3,
    #     5: 6,
    #     6: 7,
    #     7: 6,
    #     8: 3,
    #     9: 1
    # }
    # # rolls = list(product(range(1,4), repeat=3))
    # # print(len(rolls))
    # pos1 = 4
    # ans1 = 0
    # pos2 = 7
    # ans2 = 0
    # turn_val_seq = []
    # play_1 = 0
    # play_2 = 0
    # memo_dict = dict()
    # # print(tree_gamed(pos1,ans1,pos2,ans2,turn_val_seq,play_1,play_2,memo_dict))
    # print(max(tree_gamed2(pos1,ans1,pos2,ans2,memo_dict)))
    
    #PROCESS DAY 20
    # input_img, enhancement_algo = process_depth(data)
    # print(apply_enhancement(input_img,enhancement_algo,50))

    #PROCESS DAY 19
    # scanners = proccessScanners(data)
    # scanners.remove([])
    # print(scanners)
    # compareDist(scanners[0],scanners[1])
    # find_overlaps(scanners[0],scanners[1])
    # print(find_and_normalize(scanners))
    
    #PROCESS DAY 18
    # snailNums = proccessSnailNums(data)
    # # ans = snailNums[0]
    # # for i in range(1,len(snailNums)):
    #     # print(i)
    #     # ans = reduceSnailNums(addSnailNums(ans,snailNums[i]))
    # largest = 0
    # ind = []
    # for i in range(len(snailNums)):
        
    #     for j in range(i,len(snailNums)):
    #         if i != j:
    #             first = deepcopy(snailNums[i])
    #             second = deepcopy(snailNums[j])
    #             # print(first,second)
    #             new = magnitude(reduceSnailNums(addSnailNums(first,second)))
    #             # print(reduceSnailNums(addSnailNums(snailNums[i],snailNums[j])))
    #             # new = magnitude(reduceSnailNums(addSnailNums(snailNums[i],snailNums[j])))
    #             # print(snailNums[i],snailNums[j],largest,new)
    #             if new > largest:
    #                 largest = new
    #                 ind = [i,j]
    #             first = deepcopy(snailNums[i])
    #             second = deepcopy(snailNums[j])
    #             new = magnitude(reduceSnailNums(addSnailNums(second,first)))
    #             if new > largest:
    #                 largest = new
    #                 ind = [i,j]

    # print(largest)
    # print(snailNums[ind[0]],snailNums[ind[1]])
    # print(magnitude(ans))

    # print(snailNums)
    # print(reduceSnailNums(addSnailNums(a,b)))
    
    #PROCESS DAY 17
    #target area: x=25..67, y=-260..-200
    # x_window = (25,67)
    # y_window = (-260,-200)
    # found = trajectory_count(x_window,y_window)
    # print(len(found))
    # ans = set()
    # for line in data:
    #     for pair in line.split(' ')[:-1]:
    #         pair.replace(' ','')
    #         if pair != '':
    #             inter = pair.split(',')
    #             ans.add((int(inter[0]),int(inter[1])))
    # print(list(ans - set(found)))
    # print(xHit(6,x_window,11))

    #PRCOESS DAY 16
    # inputHex = '38006F45291200'
    # inputHex = '8A004A801A8002F478'
    # inputHex = '620080001611562C8802118E34'
    # inputHex = 'A0016C880162017C3686B18A3D4780'
    # inputHex = data[0]
    # for i in data:
    #     print(i[:-1])
    #     print(int(binDecode(hextobin(i[:-1]))[0],2))
    # inputHex = 'D2FE28'
    # print(data[0])
    # print(int(binDecode(hextobin(data[0]))[0],2))

    #PROCESS DAY 15
    
    # cave = processCaveHole(data)
    # with open ("test2.txt", "r") as myfile:
    #     data=myfile.readlines()
    # cave2 = processCaveHole(data)
    # cave2 = expandCave(cave2)
    # print(cave2)
    # for i in range(len(cave)):
    #     if cave[i] != cave2[i]:
    #         print(cave[i],cave2[i])

    # cave = [[1,3,5],[2,4,7]]
    # cave = expandCave(cave)
    # print(findPathGraphed(cave))
    # print(cave)
    # print(findPath(cave))
    #PROCESS DAY 14
    # start,template = procPolymer(data)
    # print(start,template)
    # # print(oneStepPoly(start,template))
    # # print(polyBuild(start,template,40))
    # print(polyCounter(start,template,40))



    #PROCESS DAY 13
    # dots,lines = procTransPaper(data)
    # print(dots,lines)
    # # print(fold(dots,lines[0]))
    # dotLS = allFolds(dots,lines)
    # print(dotLS)
    # visualizeDots(dotLS)


    #PROCESS DAY 12
    # pathLs = processPath(data)
    # print(pathLs)
    # # print(findPath(pathLs))
    # # print(DFSglo(pathLs))
    # print(DFSglo(pathLs))

    #PROCESS DAY 11
    # octopi = processOctopi(data)
    # print(octopi)
    # print(firstSimultaneous(octopi))

    #PROCESS DAY 10
    # data = processSynError(data)
    # print(data)
    # print(corruptFinder(data))

    #PROCESS DAY 9
    # heightMap = processHieghtMap(data)
    # print(heightMap)
    # print(findBasin(heightMap))
    # print(basinSearch(heightMap,(1,1)))
    # print(lowPoints(hieghtMap))

    #PROCESS DAY 8
    # outVal = processingOutputVal(data)
    # digits = procInputVal(data)
    # print(ls)
    # print(evalAllSets(digits,outVal))
    # print(uniqueSegNumbers(ls))
    

    #PROCESSING DAY 7
    # ls = processHoriz(data)
    # print(ls)
    # center = variableCenter(ls)
    # print(sum(ls)/len(ls))
    # print(center)
    # ls = [10,0,2,50,0,0,0]
    # ans = []
    # for i in range(center-10,center+10):
    #     varCenter = variableFuelCalc(ls,i)
    #     ans.append(varCenter)
    #     print(i,": ",varCenter)
    # ans.sort()
    # print(ans)
    # print(round(sum(ls)/len(ls)))
    # print(cheapestHoriz(ls))
    # print(fuelCalc(ls))

    #PROCESSING DAY 6
    # fishList = processFish(data)
    # for i in range(0,32):
    #     print(i,fishdayCount([6,8],i))
    # print(fishdayCount(fishList,19))
    # print(fishList)
    # print(fishMovingList(fishList,256))
    # print(fishListRecurs([0],18))
    # print(len(fishListRecurs(fishList,160)))

    #PROCESSING DAY 5
    # noDiag = False
    # segments,X,Y = processVents(data,noDiag)
    # vents = ventMap(segments,X,Y)
    # print(vents)
    # print(ventUnsafeCount(vents))

    #PROCESSING FOR DAY 4
    # boards,nums = processBingo(data)
    # print(lastBingoBoard(nums,boards))
        

    #PROCESSING FOR DAY 3
    # data = [x[:-1] for x in data]
    # gam = Gamma(data)
    # print(gam)
    # print(int(gam,2)) #switch from binary
    # ep = eps(gam)
    # print(ep)
    # print(int(ep,2)) 
    # print(int(gam,2)*int(ep,2))
    # print(int(oxygen(data),2))
    # print(int(co2(data),2))
    # print(int(oxygen(data),2)*int(co2(data),2))

    #PROCESSING FOR DAY 2
    # instruc = []
    # for line in data:
    #     instruc.append(line[:-1].split(' '))

