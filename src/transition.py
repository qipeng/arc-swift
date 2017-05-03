"""
Implementation of transition systems.

The TransitionSystem class is an "interface" for all of the
subclasses that are being used, but isn't really used anywhere
explicitly itself.
"""

class TransitionSystem(object):
    def __init__(self, mappings, invmappings):
        self.mappings, self.invmappings = mappings, invmappings

    def _preparetransitionset(self, parserstate):
        """ Prepares the set of gold transitions given a parser state """
        raise NotImplementedError()

    def advance(self, parserstate, action):
        """ Advances a parser state given an action """
        raise NotImplementedError()

    def goldtransition(self, parserstate, goldrels):
        """ Returns the next gold transition given the set of gold arcs """
        raise NotImplementedError()

    def trans_to_str(self, transition, state, pos, fpos=None):
        raise NotImplementedError()

    @classmethod
    def trans_from_line(self, line):
        raise NotImplementedError()

    @classmethod
    def actions_list(self):
        raise NotImplementedError()

class ArcSwift(TransitionSystem):
    @classmethod
    def actions_list(self):
        return ['Shift', 'Left-Arc', 'Right-Arc']

    def _preparetransitionset(self, parserstate):
        SHIFT = self.mappings['action']['Shift']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']

        stack, buf, head = parserstate.stack, parserstate.buf, parserstate.head

        t = []

        if len(buf) > 1:
            t += [(SHIFT, -1)]

        left_possible = False
        if len(buf) > 0:
            for si in xrange(len(stack) - 1):
                if head[stack[si]][0] < 0:
                    t += [(LEFTARC, si)]
                    left_possible = True
                    break
        if len(buf) > 1 or (len(buf) == 1 and not left_possible):
            for si in xrange(len(stack)):
                t += [(RIGHTARC, si)]
                if head[stack[si]][0] < 0:
                    break

        parserstate._transitionset = t

    def advance(self, parserstate, action):
        SHIFT = self.mappings['action']['Shift']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']

        RELS = len(self.mappings['rel'])
        cand = parserstate.transitionset()

        if isinstance(action, int):
            a, rel = self.tuple_trans_from_int(cand, action)
        else:
            rel = action[-1]
            a = action[:-1]

        stack = parserstate.stack
        buf = parserstate.buf

        if a[0] == SHIFT:
            parserstate.stack = [buf[0]] + stack
            parserstate.buf = buf[1:]
        elif a[0] == LEFTARC:
            si = a[1]
            parserstate.head[stack[si]] = [buf[0], rel]
            parserstate.stack = stack[(si+1):]
        elif a[0] == RIGHTARC:
            si = a[1]
            parserstate.head[buf[0]] = [stack[si], rel]
            parserstate.stack = [buf[0]] + stack[si:]
            parserstate.buf = buf[1:]

        self._preparetransitionset(parserstate)

    def goldtransition(self, parserstate, goldrels=None):
        SHIFT = self.mappings['action']['Shift']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']

        goldrels = goldrels or parserstate.goldrels
        stack = parserstate.stack
        buf = parserstate.buf
        head = parserstate.head

        j = buf[0]
        addedArc = False
        for n in xrange(len(stack)):
            if stack[n] in goldrels[j]:
                rel = goldrels[j][stack[n]]
                a = (LEFTARC, n, rel)
                addedArc = True
                break
            elif j in goldrels[stack[n]]:
                rel = goldrels[stack[n]][j]
                a = (RIGHTARC, n, rel)
                addedArc = True
                break
            if head[stack[n]][0] < 0: break

        if not addedArc:
            a = (SHIFT, -1, -1)

        return a

    def trans_to_str(self, t, state, pos, fpos=None):
        SHIFT = self.mappings['action']['Shift']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']

        if t[0] == SHIFT:
            if fpos is None:
                return "Shift\t%s" % (pos[state.buf[0]])
            else:
                return "Shift\t%s\t%s" % (pos[state.buf[0]], fpos[state.buf[0]])
        elif t[0] == LEFTARC:
            return "Left-Arc\t%d\t%s" % (t[1]+1, self.invmappings['rel'][t[2]])
        elif t[0] == RIGHTARC:
            if fpos is None:
                return "Right-Arc\t%d\t%s\t%s" % (t[1]+1, self.invmappings['rel'][t[2]], pos[state.buf[0]])
            else:
                return "Right-Arc\t%d\t%s\t%s\t%s" % (t[1]+1, self.invmappings['rel'][t[2]], pos[state.buf[0]], fpos[state.buf[0]])

    @classmethod
    def trans_from_line(self, line):
        if line[0] == 'Left-Arc':
            fields = { 'action':line[0], 'n':int(line[1]), 'rel':line[2] }
        elif line[0] == 'Right-Arc':
            fields = { 'action':line[0], 'n':int(line[1]), 'rel':line[2], 'pos':line[3] }
            if len(line) > 4:
                fields['fpos'] = line[4]
        elif line[0] == 'Shift':
            fields = { 'action':line[0], 'pos':line[1] }
            if len(line) > 2:
                fields['fpos'] = line[2]
        else:
            raise ValueError(line[0])
        return fields

    def tuple_trans_to_int(self, cand, t):
        SHIFT = self.mappings['action']['Shift']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']

        RELS = len(self.mappings['rel'])

        base = 0
        if t[0] == SHIFT:
            return 0

        if cand[0][0] == SHIFT:
            base = 1

        if t[0] == LEFTARC:
            return base + t[2]

        if len(cand) > 1 and cand[1][0] == LEFTARC:
            base += RELS

        if t[0] == RIGHTARC:
            return base + t[1]*RELS + t[2]

    def tuple_trans_from_int(self, cand, action):
        SHIFT = self.mappings['action']['Shift']
        RELS = len(self.mappings['rel'])
        rel = -1

        if cand[0][0] == SHIFT:
            if action == 0:
                a = cand[0]
            else:
                a = cand[(action - 1) / RELS + 1]
                rel = (action - 1) % RELS
        else:
            a = cand[action / RELS]
            rel = action % RELS

        return a, rel

class ArcEagerReduce(TransitionSystem):
    @classmethod
    def actions_list(self):
        return ['Shift', 'Left-Arc', 'Right-Arc', 'Reduce']

    def _preparetransitionset(self, parserstate):
        SHIFT = self.mappings['action']['Shift']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']
        REDUCE = self.mappings['action']['Reduce']

        stack, buf, head = parserstate.stack, parserstate.buf, parserstate.head

        t = []

        if len(buf) > 1:
            t += [(SHIFT,)]

        if len(buf) > 0 and len(stack) > 1:
            t += [(REDUCE,)]

        left_possible = False
        if len(buf) > 0 and len(stack) > 1:
            if head[stack[0]][0] < 0:
                t += [(LEFTARC,)]
                left_possible = True

        if len(buf) > 1 or (len(buf) == 1 and not left_possible):
            t += [(RIGHTARC,)]

        parserstate._transitionset = t

    def advance(self, parserstate, action):
        SHIFT = self.mappings['action']['Shift']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']
        REDUCE = self.mappings['action']['Reduce']

        RELS = len(self.mappings['rel'])
        cand = parserstate.transitionset()

        if isinstance(action, int):
            a, rel = self.tuple_trans_from_int(cand, action)
        else:
            rel = action[-1]
            a = action[:-1]

        stack = parserstate.stack
        buf = parserstate.buf

        if a[0] == SHIFT:
            parserstate.stack = [buf[0]] + stack
            parserstate.buf = buf[1:]
        elif a[0] == LEFTARC:
            parserstate.head[stack[0]] = [buf[0], rel]
            parserstate.stack = stack[1:]
        elif a[0] == RIGHTARC:
            parserstate.head[buf[0]] = [stack[0], rel]
            parserstate.stack = [buf[0]] + stack
            parserstate.buf = buf[1:]
        elif a[0] == REDUCE:
            parserstate.stack = stack[1:]

        self._preparetransitionset(parserstate)

    def goldtransition(self, parserstate, goldrels=None):
        SHIFT = self.mappings['action']['Shift']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']
        REDUCE = self.mappings['action']['Reduce']

        goldrels = goldrels or parserstate.goldrels
        stack = parserstate.stack
        buf = parserstate.buf
        head = parserstate.head

        POS = len(self.mappings['pos'])

        j = buf[0]

        norightchildren = True
        for x in buf:
            if x in goldrels[stack[0]]:
                norightchildren = False
                break

        if stack[0] in goldrels[j]:
            rel = goldrels[j][stack[0]]
            a = (LEFTARC, rel)
        elif j in goldrels[stack[0]]:
            rel = goldrels[stack[0]][j]
            a = (RIGHTARC, rel)
        elif head[stack[0]][0] >= 0 and norightchildren:
            a = (REDUCE, -1)
        else:
            a = (SHIFT, -1)

        return a

    def trans_to_str(self, t, state, pos, fpos=None):
        SHIFT = self.mappings['action']['Shift']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']
        REDUCE = self.mappings['action']['Reduce']
        if t[0] == SHIFT:
            if fpos is None:
                return "Shift\t%s" % (pos[state.buf[0]])
            else:
                return "Shift\t%s\t%s" % (pos[state.buf[0]], fpos[state.buf[0]])
        elif t[0] == LEFTARC:
            return "Left-Arc\t%s" % (self.invmappings['rel'][t[1]])
        elif t[0] == RIGHTARC:
            if fpos is None:
                return "Right-Arc\t%s\t%s" % (self.invmappings['rel'][t[1]], pos[state.buf[0]])
            else:
                return "Right-Arc\t%s\t%s\t%s" % (self.invmappings['rel'][t[1]], pos[state.buf[0]], fpos[state.buf[0]])
        elif t[0] == REDUCE:
            return "Reduce"

    @classmethod
    def trans_from_line(self, line):
        if line[0] == 'Left-Arc':
            fields = { 'action':line[0], 'rel':line[1] }
        elif line[0] == 'Right-Arc':
            fields = { 'action':line[0], 'rel':line[1], 'pos':line[2] }
            if len(line) > 3:
                fields['fpos'] = line[3]
        elif line[0] == 'Shift':
            fields = { 'action':line[0], 'pos':line[1] }
            if len(line) > 2:
                fields['fpos'] = line[2]
        elif line[0] == 'Reduce':
            fields = { 'action':line[0] }
        else:
            raise ValueError(line[0])
        return fields

    def tuple_trans_to_int(self, cand, t):
        SHIFT = self.mappings['action']['Shift']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']
        REDUCE = self.mappings['action']['Reduce']

        RELS = len(self.mappings['rel'])

        base = 0
        if t[0] == SHIFT:
            return base

        base += 1

        if t[0] == REDUCE:
            return base

        base += 1

        if t[0] == LEFTARC:
            return base + t[1]

        base += RELS

        if t[0] == RIGHTARC:
            return base + t[1]

    def tuple_trans_from_int(self, cand, action):
        SHIFT = self.mappings['action']['Shift']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']
        REDUCE = self.mappings['action']['Reduce']
        RELS = len(self.mappings['rel'])
        rel = -1

        base = 0
        if action == base:
            a = (SHIFT,)
        base += 1

        if action == base:
            a = (REDUCE,)
        base += 1

        if base <= action < base + RELS:
            a = (LEFTARC,)
            rel = action - base
        base += RELS

        if base <= action < base + RELS:
            a = (RIGHTARC,)
            rel = action - base

        return a, rel

class ArcEagerShift(ArcEagerReduce):
    def goldtransition(self, parserstate, goldrels=None):
        SHIFT = self.mappings['action']['Shift']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']
        REDUCE = self.mappings['action']['Reduce']

        goldrels = goldrels or parserstate.goldrels
        stack = parserstate.stack
        buf = parserstate.buf
        head = parserstate.head

        POS = len(self.mappings['pos'])

        j = buf[0]

        has_right_children = False
        for i in buf:
            if i in goldrels[stack[0]]:
                has_right_children = True
                break

        must_reduce = False
        for i in stack:
            if i in goldrels[j] or j in goldrels[i]:
                must_reduce = True
                break
            if head[i][0] < 0:
                break

        if stack[0] in goldrels[j]:
            rel = goldrels[j][stack[0]]
            a = (LEFTARC, rel)
        elif j in goldrels[stack[0]]:
            rel = goldrels[stack[0]][j]
            a = (RIGHTARC, rel)
        elif not must_reduce or head[stack[0]][0] < 0 or has_right_children:
            a = (SHIFT, -1)
        else:
            a = (REDUCE, -1)

        return a

class ArcStandard(TransitionSystem):
    @classmethod
    def actions_list(self):
        return ['Shift', 'Left-Arc', 'Right-Arc']

    def _preparetransitionset(self, parserstate):
        SHIFT = self.mappings['action']['Shift']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']

        stack, buf, head = parserstate.stack, parserstate.buf, parserstate.head

        t = []

        if len(buf) > 0:
            t += [(SHIFT,)]

        if len(stack) > 2:
            t += [(LEFTARC,)]

        if len(stack) > 1:
            t += [(RIGHTARC,)]

        parserstate._transitionset = t

    def advance(self, parserstate, action):
        SHIFT = self.mappings['action']['Shift']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']

        RELS = len(self.mappings['rel'])
        cand = parserstate.transitionset()

        if isinstance(action, int):
            a, rel = self.tuple_trans_from_int(cand, action)
        else:
            rel = action[-1]
            a = action[:-1]

        stack = parserstate.stack
        buf = parserstate.buf

        if a[0] == SHIFT:
            parserstate.stack = [buf[0]] + stack
            parserstate.buf = buf[1:]
        elif a[0] == LEFTARC:
            parserstate.head[stack[1]] = [stack[0], rel]
            parserstate.stack = [stack[0]] + stack[2:]
        elif a[0] == RIGHTARC:
            parserstate.head[stack[0]] = [stack[1], rel]
            parserstate.stack = stack[1:]

        self._preparetransitionset(parserstate)

    def goldtransition(self, parserstate, goldrels=None):
        SHIFT = self.mappings['action']['Shift']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']

        goldrels = goldrels or parserstate.goldrels
        stack = parserstate.stack
        buf = parserstate.buf
        head = parserstate.head

        POS = len(self.mappings['pos'])

        stack0_done = True
        for x in buf:
            if x in goldrels[stack[0]]:
                stack0_done = False
                break

        if len(stack) > 2 and stack[1] in goldrels[stack[0]]:
            rel = goldrels[stack[0]][stack[1]]
            a = (LEFTARC, rel)
        elif len(stack) > 1 and stack[0] in goldrels[stack[1]] and stack0_done:
            rel = goldrels[stack[1]][stack[0]]
            a = (RIGHTARC, rel)
        else:
            a = (SHIFT, -1)

        return a

    def trans_to_str(self, t, state, pos, fpos=None):
        SHIFT = self.mappings['action']['Shift']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']
        if t[0] == SHIFT:
            if fpos is None:
                return "Shift\t%s" % (pos[state.buf[0]])
            else:
                return "Shift\t%s\t%s" % (pos[state.buf[0]], fpos[state.buf[0]])
        elif t[0] == LEFTARC:
            return "Left-Arc\t%s" % (self.invmappings['rel'][t[1]])
        elif t[0] == RIGHTARC:
            return "Right-Arc\t%s" % (self.invmappings['rel'][t[1]])

    @classmethod
    def trans_from_line(self, line):
        if line[0] == 'Left-Arc':
            fields = { 'action':line[0], 'rel':line[1] }
        elif line[0] == 'Right-Arc':
            fields = { 'action':line[0], 'rel':line[1] }
        elif line[0] == 'Shift':
            fields = { 'action':line[0], 'pos':line[1] }
            if len(line) > 2:
                fields['fpos'] = line[2]
        else:
            raise ValueError(line[0])
        return fields

    def tuple_trans_to_int(self, cand, t):
        SHIFT = self.mappings['action']['Shift']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']

        RELS = len(self.mappings['rel'])

        base = 0
        if t[0] == SHIFT:
            return base

        base += 1

        if t[0] == LEFTARC:
            return base + t[1]

        base += RELS

        if t[0] == RIGHTARC:
            return base + t[1]

    def tuple_trans_from_int(self, cand, action):
        SHIFT = self.mappings['action']['Shift']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']
        RELS = len(self.mappings['rel'])
        rel = -1

        base = 0
        if action == base:
            a = (SHIFT,)
        base += 1

        if base <= action < base + RELS:
            a = (LEFTARC,)
            rel = action - base
        base += RELS

        if base <= action < base + RELS:
            a = (RIGHTARC,)
            rel = action - base

        return a, rel

class ArcHybrid(ArcStandard):
    def _preparetransitionset(self, parserstate):
        SHIFT = self.mappings['action']['Shift']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']

        stack, buf, head = parserstate.stack, parserstate.buf, parserstate.head

        t = []

        if len(buf) > 0:
            t += [(SHIFT,)]

        if len(buf) > 0 and len(stack) > 1 and head[stack[0]][0] < 0:
            t += [(LEFTARC,)]

        if len(stack) > 1 and head[stack[0]][0] < 0:
            t += [(RIGHTARC,)]

        parserstate._transitionset = t

    def advance(self, parserstate, action):
        SHIFT = self.mappings['action']['Shift']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']

        RELS = len(self.mappings['rel'])
        cand = parserstate.transitionset()

        if isinstance(action, int):
            a, rel = self.tuple_trans_from_int(cand, action)
        else:
            rel = action[-1]
            a = action[:-1]

        stack = parserstate.stack
        buf = parserstate.buf

        if a[0] == SHIFT:
            parserstate.stack = [buf[0]] + stack
            parserstate.buf = buf[1:]
        elif a[0] == LEFTARC:
            parserstate.head[stack[0]] = [buf[0], rel]
            parserstate.stack = stack[1:]
        elif a[0] == RIGHTARC:
            parserstate.head[stack[0]] = [stack[1], rel]
            parserstate.stack = stack[1:]

        self._preparetransitionset(parserstate)

    def goldtransition(self, parserstate, goldrels=None):
        SHIFT = self.mappings['action']['Shift']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']

        goldrels = goldrels or parserstate.goldrels
        stack = parserstate.stack
        buf = parserstate.buf
        head = parserstate.head

        POS = len(self.mappings['pos'])

        stack0_done = True
        for x in buf:
            if x in goldrels[stack[0]]:
                stack0_done = False
                break

        if len(buf) > 0 and stack[0] in goldrels[buf[0]]:
            rel = goldrels[buf[0]][stack[0]]
            a = (LEFTARC, rel)
        elif len(stack) > 1 and stack[0] in goldrels[stack[1]] and stack0_done:
            rel = goldrels[stack[1]][stack[0]]
            a = (RIGHTARC, rel)
        else:
            a = (SHIFT, -1)

        return a
