module.exports = {
  title: `Oha's`,
  description: `하성민의 개발여행`,
  language: `ko`, // `ko`, `en` => currently support versions for Korean and English
  siteUrl: `https://xman227.github.io`,
  ogImage: `/og-image.png`, // Path to your in the 'static' folder
  comments: {
    utterances: {
        repo: '' // zoomkoding/zoomkoding-gatsby-blog
    },
  },
  comments: { 
    utterances: {
      repo: `xman227/blog_comments`, // `zoomkoding/zoomkoding-gatsby-blog`,
    },
  },
  ga: '0', // Google Analytics Tracking ID
  author: {
    name: `하성민`,
    bio: {
      role: `개발자`,
      description: ['행복을 보태는', '코드를 노래하는', 'AI를 사귀는 '],
      thumbnail: 'memo.gif', // Path to the image in the 'asset' folder
    },
    social: {
      github: `https://github.com/xman227`, // `https://github.com/xman227`,
      linkedIn: ``, // `https://www.linkedin.com/in/jinhyeok-jeong-800871192`,
      email: `x22z@naver.com`, // `zoomkoding@gmail.com`,
    },
  },

  // metadata for About Page
  about: {
    timestamps: [
      // =====       [Timestamp Sample and Structure]      =====
      // ===== 🚫 Don't erase this sample (여기 지우지 마세요!) =====
      {
        date: '',
        activity: '',
        links: {
          github: '',
          post: '',
          googlePlay: '',
          appStore: '',
          demo: '',
        },
      },
      // ========================================================
      // ========================================================

      {
        date: '2017.03 ~',
        activity: '인하대학교 경영학과 입학',
        links: {
          post: '',
          github: '',
          demo: '',
        },
      },


      {
        date: '2018.12 ~',
        activity: 'Fitness 네이버 블로그 운영',
        links: {
          post: 'https://blog.naver.com/bulkup-star_maybe',
          github: '',
          demo: '',
        },
      },
      {
        date: '2021.12 ~',
        activity: '모두의 연구소 교육기관 Aiffel',
        links: {
          post: '',
          github: '',
          demo: '',
        },
      },

      {
        date: '2022.02 ~',
        activity: '현 블로그 개발 및 운영',
        links: {
          post: '',
          github: '',
          demo: '',
        },
      }



    ],

    projects: [
      // =====        [Project Sample and Structure]        =====
      // ===== 🚫 Don't erase this sample (여기 지우지 마세요!)  =====
      {
        title: '',
        description: '',
        techStack: ['', ''],
        thumbnailUrl: '',
        links: {
          post: '',
          github: '',
          googlePlay: '',
          appStore: '',
          demo: '',
        },
      },
      // ========================================================
      // ========================================================
      {
        title: 'Aiffel Exploration Project 진행',
        description:
          '인공지능 교육기관인 Aiffel 에서 Deep learning 모델을 활용해 고양이 분류기, 자동 가사 생성기, 이미지 생성기 등을 제작하며 많은 실력과 지식을 쌓았습니다. 추후 NLP 노드 분야로 나아가 헬스케어 분야에 접목하는 것이 목표입니다. Git hub 에 관련 공부자료를 업로드하고 있습니다.',
        techStack: ['python', 'scikit learn', 'tensorflow'],
        thumbnailUrl: 'aiffel.PNG',
        links: {
          post: '',
          github: 'https://github.com/xman227/Aiffel_oHA',
          demo: 'https://github.com/xman227/Aiffel_oHA',
        },
      },

      // ========================================================
      // ========================================================
      {
        title: 'Aiffel Exploration Project 진행',
        description:
          '개인적인 Fitness Blog, Oha 를 운영하고 있습니다. 헬스케어와 인공지능을 결합해 더 나은 삶을 만드는 것이 목표입니다. 모두가 건강한 몸과 마음을 만들 수 있도록 노력하고 있습니다.',
        techStack: ['Bulk up', 'Diat', 'Healthy life'],
        thumbnailUrl: 'fitnessblog.PNG',
        links: {
          post: 'https://blog.naver.com/bulkup-star_maybe', //'/gatsby-starter-zoomkoding-introduction' 로 홈페이지 내 경로 설정 가능
          github: '',
          demo: 'https://blog.naver.com/bulkup-star_maybe',
        },
      }
    ],
  },
};
