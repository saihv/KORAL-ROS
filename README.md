----------------
# KORAL_ROS
----------------

KORAL_ROS is a ROS based computer vision pipeline, which combines GPU based feature detection, description and matching. 

This repository serves as an extension to [KORAL](https://github.com/komrad36/KORAL), an extremely fast, highly accurate, 
scale- and rotation-invariant CPU-GPU cooperative detector-descriptor that uses FAST
keypoints and LATCH descriptors, combining it with [CUDAK2NN](https://github.com/komrad36/CUDAK2NN), a super-fast GPU implementation 
of a brute-force matcher for 512-bit binary descriptors, both originally developed by 
[Kareem Omar](https://github.com/komrad36). All credits for these amazingly fast kernels go to the original author(s).

In this repository, KORAL and CUDAK2NN have been adapted into a real time framework, where images are read
in succession on which detection and matching is performed. The sample code in koral_node.cpp subscribes to two ROS topics
'imageL' and 'imageR', simulating left and right camera views, performs feature extraction on 
both images and brute force matching between the two views. This is aimed at being a starting
point for GPU based vision algorithms for autonomous vehicles.

**Dependencies:**

* AVX2 capable CPU
* CUDA capable GPU
* ROS
* OpenCV (for retrieving keypoints and matches)

This code is meant only as an example to get started with, as numerous improvements can be 
made to the current functionality (example: asynchronous detection for multiple images).

**Sample benchmark:** (i7-6770HQ, GTX 1080)

| Image resolution        | Detection (ms per image)           | Matching (ms)  |
| :-------------: |:-------------:| :-----:|
| 640x480      | 3 | 1 |
| 1920x1080      | 7.5      |   5 |

# Licenses #
**KORAL is licensed under the MIT License : https://opensource.org/licenses/mit-license.php**

**Copyright(c) 2016 Kareem Omar, Christopher Parker**

Permission is hereby granted, free of charge,
to any person obtaining a copy of this software and associated documentation
files(the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and / or sell copies of the Software, and to permit persons to whom
the Software is furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


Note again that KORAL is a work in progress.
Suggestions and improvements are welcomed.

- - - -

The FAST detector was created by Edward Rosten and Tom Drummond
as described in the 2006 paper by Rosten and Drummond:
"Machine learning for high-speed corner detection"
        Edward Rosten and Tom Drummond
https://www.edwardrosten.com/work/rosten_2006_machine.pdf

**The FAST detector is BSD licensed:**

**Copyright(c) 2006, 2008, 2009, 2010 Edward Rosten**
**All rights reserved.**

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met :


*Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

*Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and / or other materials provided with the distribution.

*Neither the name of the University of Cambridge nor the names of
its contributors may be used to endorse or promote products derived
from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES(INCLUDING, BUT NOT LIMITED TO,
	PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
	PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
	LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING
		NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


- - - -
